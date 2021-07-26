// Copyright 2019-2021 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <charconv>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ygm/detail/mpi.hpp>
#include <ygm/detail/ygm_cereal_archive.hpp>
#include <ygm/meta/functional.hpp>
#define test_buffer_capacity 1048576

namespace ygm {

class comm::impl {
public:
  impl(MPI_Comm c, int buffer_capacity = 16 * 1024) {
    ASSERT_MPI(MPI_Comm_dup(c, &m_comm_async));
    // Large messages use a different communicator to a separate listener thread
    ASSERT_MPI(MPI_Comm_dup(c, &m_comm_large_async));
    ASSERT_MPI(MPI_Comm_dup(c, &m_comm_barrier));
    ASSERT_MPI(MPI_Comm_dup(c, &m_comm_other));
    ASSERT_MPI(MPI_Comm_size(m_comm_async, &m_comm_size));
    ASSERT_MPI(MPI_Comm_rank(m_comm_async, &m_comm_rank));
    m_buffer_capacity = buffer_capacity;

    // Get node/core indices
    init_local_remote_comms();

    // Allocate intermediate send buffers
    for (int i = 0; i < m_comm_local_size; ++i) {
      intermediate_send_buffers.push_back(allocate_buffer());
    }

    // Allocate final send buffers
    for (int i = 0; i < m_comm_remote_size; ++i) {
      final_send_buffers.push_back(allocate_buffer());
    }

    // launch listener threads
    m_large_listener = std::thread(&impl::listen_large, this);
    m_local_listener = std::thread(&impl::listen_local, this);
    m_remote_listener = std::thread(&impl::listen_remote, this);
  }

  ~impl() {
    barrier();
    // send kill signal to self (large listener thread)
    MPI_Send(NULL, 0, MPI_BYTE, m_comm_rank, 0, m_comm_large_async);
    // send kill signal to self (local listener thread)
    MPI_Send(NULL, 0, MPI_BYTE, m_comm_local_rank, 0, m_comm_local);
    // send kill signal to self (remote listener thread)
    MPI_Send(NULL, 0, MPI_BYTE, m_comm_remote_rank, 0, m_comm_remote);

    // Join listener threads.
    m_large_listener.join();
    m_local_listener.join();
    m_remote_listener.join();
    // Free cloned communicator.
    ASSERT_RELEASE(MPI_Barrier(m_comm_async) == MPI_SUCCESS);
    MPI_Comm_free(&m_comm_async);
    MPI_Comm_free(&m_comm_large_async);
    MPI_Comm_free(&m_comm_barrier);
    MPI_Comm_free(&m_comm_other);
    MPI_Comm_free(&m_comm_local);
    MPI_Comm_free(&m_comm_remote);
  }

  int size() const { return m_comm_size; }
  int rank() const { return m_comm_rank; }
  int local_size() const { return m_comm_local_size; }
  int local_rank() const { return m_comm_local_rank; }
  int remote_size() const { return m_comm_remote_size; }
  int remote_rank() const { return m_comm_remote_rank; }

  template <typename... SendArgs>
  void async(int dest, const SendArgs &...args) {
    ASSERT_DEBUG(dest < m_comm_size);
    if (dest == m_comm_rank) {
      local_receive(std::forward<const SendArgs>(args)...);
    } else {
      m_send_count++;
      auto dest_index = find_lr_indices(dest);
      std::vector<char> data =
          pack_lambda(std::forward<const SendArgs>(args)...);
      m_local_bytes_sent += data.size();

      if (data.size() + sizeof(header_t) < m_buffer_capacity) {
        // check if buffer doesn't have enough space
        auto header = pack_header(rank(), dest_index.second, data.size());
        header_t *hdr = reinterpret_cast<header_t *>(&header);
        // check if buffer doesn't have enough space
        if (data.size() + header.size() +
                intermediate_send_buffers[dest_index.first]->size() >
            m_buffer_capacity - sizeof(char)) {
          async_inter_flush(dest_index.first);
        }
        // insert header followed by data to intermediate destination(core
        // offset)
        intermediate_send_buffers[dest_index.first]->insert(
            intermediate_send_buffers[dest_index.first]->end(), header.begin(),
            header.end());
        intermediate_send_buffers[dest_index.first]->insert(
            intermediate_send_buffers[dest_index.first]->end(), data.begin(),
            data.end());

      } else { // Large message
        send_large_message(data, dest);
      }
    }
    // check if intermediate listener has queued transits to process
    if (transit_queue_peek_size() > 0) {
      transit_queue_process();
    }
    // check if final listener has queued arrivals to process
    if (arrival_queue_peek_size() > 0) {
      arrival_queue_process();
    }
  }

  // Will move this somewhere cleaner.
  struct header_t {
    uint64_t src : 16;
    uint64_t dst : 16;
    uint64_t len : 32;
  } __attribute__((packed));

  std::vector<char> pack_header(int src, int dest, int data_size) {
    /*Probably there is a more efficient way to do this*/

    header_t hdr_struct{(uint64_t)src, (uint64_t)dest, (uint64_t)data_size};

    auto ptr = reinterpret_cast<const char *>(&hdr_struct);
    auto hdr = std::vector<char>(ptr, ptr + sizeof(header_t));
    return hdr;
  }

  // //
  // // Blocking barrier
  // void barrier() {
  //   int64_t all_count = -1;
  //   while (all_count != 0) {
  //     receive_queue_process();
  //     do {
  //       async_flush_all();
  //       std::this_thread::yield();
  //     } while (receive_queue_process());

  //     int64_t local_count = m_send_count - m_recv_count;

  //     ASSERT_MPI(MPI_Allreduce(&local_count, &all_count, 1, MPI_INT64_T,
  //                              MPI_SUM, m_comm_barrier));
  //     std::this_thread::yield();
  //     // std::cout << "MPI_Allreduce() " << std::endl;
  //   }
  // }

  void wait_local_idle() {
    transit_queue_process();
    do {
      async_inter_flush_all();
      std::this_thread::yield();
    } while (transit_queue_process());

    arrival_queue_process();
    do {
      async_final_flush_all();
      std::this_thread::yield();
    } while (arrival_queue_process());
  }

  void barrier() {
    while (true) {
      wait_local_idle();
      MPI_Request req = MPI_REQUEST_NULL;
      int64_t first_all_count{-1};
      // int64_t first_local_count = m_send_count - m_recv_count;
      // The above line makes sure send and recv count add up but I wonder if I
      // should also include send and recv of first hop. I don't think so.
      int64_t first_local_count =
          m_send_count + inter_send_count - inter_recv_count - m_recv_count;
      ASSERT_MPI(MPI_Iallreduce(&first_local_count, &first_all_count, 1,
                                MPI_INT64_T, MPI_SUM, m_comm_barrier, &req));

      while (true) {
        int test_flag{-1};
        ASSERT_MPI(MPI_Test(&req, &test_flag, MPI_STATUS_IGNORE));
        if (test_flag) {
          if (first_all_count == 0) {
            // double check
            int64_t second_all_count{-1};
            int64_t second_local_count = m_send_count + inter_send_count -
                                         inter_recv_count - m_recv_count;
            ASSERT_MPI(MPI_Allreduce(&second_local_count, &second_all_count, 1,
                                     MPI_INT64_T, MPI_SUM, m_comm_barrier));
            if (second_all_count == 0) {
              ASSERT_RELEASE(first_local_count == second_local_count);
              return;
            }
          }
          break; // failed, start over
        } else {
          wait_local_idle();
        }
      }
    }
  }

  // //  SOMETHING WRONG :(
  // // Non-blocking barrier loop
  // void barrier() {
  //   std::pair<int64_t, int64_t> last{-1, -2}, current{-3, -4}, local{-5,
  //   -6}; MPI_Request req = MPI_REQUEST_NULL;

  //   do {
  //     receive_queue_process();
  //     do { async_flush_all(); } while (receive_queue_process());

  //     int64_t local_count = m_send_count - m_recv_count;

  //     if (req == MPI_REQUEST_NULL) {
  //       last = current;
  //       current = {-3, -4};
  //       local = std::make_pair(m_send_count, m_recv_count);
  //       ASSERT_MPI(MPI_Iallreduce(&local, &current, 2, MPI_INT64_T,
  //       MPI_SUM,
  //                                 m_comm_barrier, &req));
  //     } else {
  //       int flag{-1};
  //       ASSERT_MPI(MPI_Test(&req, &flag, MPI_STATUS_IGNORE));
  //       if (flag) {
  //         req = MPI_REQUEST_NULL;
  //       } else {
  //         std::this_thread::yield();
  //       }
  //     }
  //   } while (req != MPI_REQUEST_NULL || current.first != current.second ||
  //            last != current);
  //   ASSERT_MPI(MPI_Barrier(m_comm_barrier));
  // }

  void async_flush(int dest) {
    if (dest != m_comm_rank) {
      // Skip dest == m_comm_rank;   Only kill messages go to self.
      if (m_vec_send_buffers[dest]->size() == 0)
        return;
      auto buffer = allocate_buffer();
      std::swap(buffer, m_vec_send_buffers[dest]);
      ASSERT_MPI(MPI_Send(buffer->data(), buffer->size(), MPI_BYTE, dest, 0,
                          m_comm_async));
      free_buffer(buffer);
    }
  }

  void async_inter_flush(int index) {
    if (index != m_comm_local_rank) {
      if (intermediate_send_buffers[index]->size() == 0)
        return;
      auto buffer = allocate_buffer();
      intermediate_send_buffers[index]->push_back(0); // Nullterminating buffer
      std::swap(buffer, intermediate_send_buffers[index]);
      ASSERT_MPI(MPI_Send(buffer->data(), buffer->size(), MPI_BYTE, index, 0,
                          m_comm_local));
      free_buffer(buffer);
    }
  }

  void async_final_flush(int index) {
    if (index != m_comm_remote_rank) {
      if (final_send_buffers[index]->size() == 0)
        return;
      auto buffer = allocate_buffer();
      final_send_buffers[index]->push_back(0); // Nullterminating buffer
      std::swap(buffer, final_send_buffers[index]);
      ASSERT_MPI(MPI_Send(buffer->data(), buffer->size(), MPI_BYTE, index, 0,
                          m_comm_remote));
      free_buffer(buffer);
    } else { // For local messages, we copy the buffer and add it to the arrival
             // queue
      if (final_send_buffers[index]->size() == 0)
        return;
      auto buffer = allocate_buffer();
      buffer->assign(final_send_buffers[index]->begin(),
                     final_send_buffers[index]->end());
      buffer->push_back(0);
      arrival_queue_push_back(buffer, -2); //-2 indicates local message
      m_self_buffer_count = 0;             // reset counter
      free_buffer(final_send_buffers[index]);
    }
  }

  void async_flush_all() {
    for (int i = 0; i < size(); ++i) {
      int dest = (rank() + i) % size();
      async_flush(dest);
    }
    // TODO async_flush_bcast(); goes here
  }

  void async_inter_flush_all() {
    for (int i = 0; i < local_size(); ++i) {
      int dest = (local_rank() + i) % local_size();
      async_inter_flush(dest);
    }
    // TODO async_flush_bcast(); goes here
  }

  void async_final_flush_all() {
    for (int i = 0; i < remote_size(); ++i) {
      int dest = (remote_rank() + i) % remote_size();
      async_final_flush(dest);
    }
    // TODO async_flush_bcast(); goes here
  }

  int64_t local_bytes_sent() const { return m_local_bytes_sent; }

  void reset_bytes_sent_counter() { m_local_bytes_sent = 0; }

  int64_t local_rpc_calls() const { return m_local_rpc_calls; }

  void reset_rpc_call_counter() { m_local_rpc_calls = 0; }

  template <typename T> T all_reduce_sum(const T &t) const {
    T to_return;
    ASSERT_MPI(MPI_Allreduce(&t, &to_return, 1, detail::mpi_typeof(T()),
                             MPI_SUM, m_comm_other));
    return to_return;
  }

  template <typename T> T all_reduce_min(const T &t) const {
    T to_return;
    ASSERT_MPI(MPI_Allreduce(&t, &to_return, 1, detail::mpi_typeof(T()),
                             MPI_MIN, m_comm_other));
    return to_return;
  }

  template <typename T> T all_reduce_max(const T &t) const {
    T to_return;
    ASSERT_MPI(MPI_Allreduce(&t, &to_return, 1, detail::mpi_typeof(T()),
                             MPI_MAX, m_comm_other));
    return to_return;
  }

  template <typename T>
  void mpi_send(const T &data, int dest, int tag, MPI_Comm comm) const {
    std::vector<char> packed;
    cereal::YGMOutputArchive oarchive(packed);
    oarchive(data);
    size_t packed_size = packed.size();
    ASSERT_RELEASE(packed_size < 1024 * 1024 * 1024);
    ASSERT_MPI(MPI_Send(&packed_size, 1, detail::mpi_typeof(packed_size), dest,
                        tag, comm));
    ASSERT_MPI(MPI_Send(packed.data(), packed_size, MPI_BYTE, dest, tag, comm));
  }

  template <typename T> T mpi_recv(int source, int tag, MPI_Comm comm) const {
    std::vector<char> packed;
    size_t packed_size{0};
    ASSERT_MPI(MPI_Recv(&packed_size, 1, detail::mpi_typeof(packed_size),
                        source, tag, comm, MPI_STATUS_IGNORE));
    packed.resize(packed_size);
    ASSERT_MPI(MPI_Recv(packed.data(), packed_size, MPI_BYTE, source, tag, comm,
                        MPI_STATUS_IGNORE));

    T to_return;
    cereal::YGMInputArchive iarchive(packed.data(), packed.size());
    iarchive(to_return);
    return to_return;
  }

  template <typename T>
  T mpi_bcast(const T &to_bcast, int root, MPI_Comm comm) const {
    std::vector<char> packed;
    cereal::YGMOutputArchive oarchive(packed);
    if (rank() == root) {
      oarchive(to_bcast);
    }
    size_t packed_size = packed.size();
    ASSERT_RELEASE(packed_size < 1024 * 1024 * 1024);
    ASSERT_MPI(MPI_Bcast(&packed_size, 1, detail::mpi_typeof(packed_size), root,
                         comm));
    if (rank() != root) {
      packed.resize(packed_size);
    }
    ASSERT_MPI(MPI_Bcast(packed.data(), packed_size, MPI_BYTE, root, comm));

    cereal::YGMInputArchive iarchive(packed.data(), packed.size());
    T to_return;
    iarchive(to_return);
    return to_return;
  }

  /**
   * @brief Tree based reduction, could be optimized significantly
   *
   * @tparam T
   * @tparam MergeFunction
   * @param in
   * @param merge
   * @return T
   */
  template <typename T, typename MergeFunction>
  T all_reduce(const T &in, MergeFunction merge) const {
    int first_child = 2 * rank() + 1;
    int second_child = 2 * (rank() + 1);
    int parent = (rank() - 1) / 2;

    // Step 1: Receive from children, merge into tmp
    T tmp = in;
    if (first_child < size()) {
      T fc = mpi_recv<T>(first_child, 0, m_comm_other);
      tmp = merge(tmp, fc);
    }
    if (second_child < size()) {
      T sc = mpi_recv<T>(second_child, 0, m_comm_other);
      tmp = merge(tmp, sc);
    }

    // Step 2: Send merged to parent
    if (rank() != 0) {
      mpi_send(tmp, parent, 0, m_comm_other);
    }

    // Step 3:  Rank 0 bcasts
    T to_return = mpi_bcast(tmp, 0, m_comm_other);
    return to_return;
  }

private:
  /**
   * @brief Listener thread
   *
   */
  void listen_large() {
    while (true) {
      MPI_Status status;
      ASSERT_MPI(
          MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, m_comm_large_async, &status));
      if (status.MPI_SOURCE == m_comm_rank) {
        ASSERT_MPI(MPI_Recv(NULL, 0, MPI_BYTE, status.MPI_SOURCE, MPI_ANY_TAG,
                            m_comm_large_async, MPI_STATUS_IGNORE));
        break;
      }
      int tag = status.MPI_TAG;
      int src = status.MPI_SOURCE;
      int count;
      ASSERT_DEBUG(tag == large_message_tag);
      ASSERT_MPI(MPI_Get_count(&status, MPI_BYTE, &count))
      // Allocate large buffer
      auto large_recv_buff = std::make_shared<std::vector<char>>(count);
      // Receive large message
      receive_large_message(large_recv_buff, src, count);
      // Add buffer to arrival queue
      arrival_queue_push_back(large_recv_buff, src);
    }
  }

  void listen_local() {
    while (true) {
      auto recv_buffer = allocate_buffer();
      recv_buffer->resize(m_buffer_capacity); // TODO:  does this clear?
      MPI_Status status;
      ASSERT_MPI(MPI_Recv(recv_buffer->data(), m_buffer_capacity, MPI_BYTE,
                          MPI_ANY_SOURCE, MPI_ANY_TAG, m_comm_local, &status));

      int count;
      ASSERT_MPI(MPI_Get_count(&status, MPI_BYTE, &count))
      recv_buffer->resize(count);
      // Check for kill signal
      if (status.MPI_SOURCE == m_comm_local_rank)
        break;
      // Add buffer to receive queue
      transit_queue_push_back(recv_buffer, -5);
    }
  }

  void listen_remote() {
    while (true) {
      auto recv_buffer = allocate_buffer();
      recv_buffer->resize(m_buffer_capacity); // TODO:  does this clear?
      MPI_Status status;
      ASSERT_MPI(MPI_Recv(recv_buffer->data(), m_buffer_capacity, MPI_BYTE,
                          MPI_ANY_SOURCE, MPI_ANY_TAG, m_comm_remote, &status));

      int count;
      ASSERT_MPI(MPI_Get_count(&status, MPI_BYTE, &count))
      recv_buffer->resize(count);
      // Check for kill signal
      if (status.MPI_SOURCE == m_comm_remote_rank)
        break;

      // Add buffer to receive queue
      arrival_queue_push_back(recv_buffer, -10);
    }
  }

  std::pair<int, int> find_lr_indices(const int dest) {
    int remote_index = dest / m_comm_local_size;
    int local_index = dest % m_comm_local_size;

    // std::cout<<"LR:"<<dest<<"%%"<<local_index<<"**"<<remote_index<<"\n";
    auto indices = std::make_pair(local_index, remote_index);
    return indices;
  }

  /*
   * @brief Send a large message
   *
   * @param dest Destination for message
   * @param msg Packed message to send
   */
  void send_large_message(const std::vector<char> &msg, const int dest) {
    size_t size = msg.size();
    ASSERT_MPI(MPI_Send(msg.data(), size, MPI_BYTE, dest, large_message_tag,
                        m_comm_large_async));
  }

  /*
   * @brief Receive a large message that has been announced
   *
   * @param src Source of message
   * @param msg Buffer to hold message
   */
  void receive_large_message(std::shared_ptr<std::vector<char>> msg,
                             const int src, const size_t size) {
    ASSERT_MPI(MPI_Recv(msg->data(), size, MPI_BYTE, src, large_message_tag,
                        m_comm_large_async, MPI_STATUS_IGNORE));
  }

  /**
   * @brief Allocates buffer; checks free pool first.
   *
   * @return std::shared_ptr<std::vector<char>>
   */
  std::shared_ptr<std::vector<char>> allocate_buffer() {
    std::scoped_lock lock(m_vec_free_buffers_mutex);
    if (m_vec_free_buffers.empty()) {
      auto to_return = std::make_shared<std::vector<char>>();
      to_return->reserve(m_buffer_capacity);
      return to_return;
    } else {
      auto to_return = m_vec_free_buffers.back();
      m_vec_free_buffers.pop_back();
      return to_return;
    }
  }

  /**
   * @brief Frees a previously allocated buffer.  Adds buffer to free pool.
   *
   * @param b buffer to free
   */
  void free_buffer(std::shared_ptr<std::vector<char>> b) {
    b->clear();
    std::scoped_lock lock(m_vec_free_buffers_mutex);
    m_vec_free_buffers.push_back(b);
  }

  // size_t receive_queue_peek_size() const { return m_receive_queue.size(); }
  size_t transit_queue_peek_size() const { return m_transit_queue.size(); }
  size_t arrival_queue_peek_size() const { return m_arrival_queue.size(); }

  std::pair<std::shared_ptr<std::vector<char>>, int> transit_queue_try_pop() {
    std::scoped_lock lock(m_transit_queue_mutex);
    if (m_transit_queue.empty()) {
      return std::make_pair(std::shared_ptr<std::vector<char>>(), int(-1));
    } else {
      auto to_return = m_transit_queue.front();
      m_transit_queue.pop_front();
      return to_return;
    }
  }

  std::pair<std::shared_ptr<std::vector<char>>, int> arrival_queue_try_pop() {
    std::scoped_lock lock(m_arrival_queue_mutex);
    if (m_arrival_queue.empty()) {
      return std::make_pair(std::shared_ptr<std::vector<char>>(), int(-1));
    } else {
      auto to_return = m_arrival_queue.front();
      m_arrival_queue.pop_front();
      return to_return;
    }
  }

  void transit_queue_push_back(std::shared_ptr<std::vector<char>> b, int from) {
    size_t current_size = 0;
    {
      std::scoped_lock lock(m_transit_queue_mutex);
      m_transit_queue.push_back(std::make_pair(b, from));
      current_size = m_transit_queue.size();
    }
    if (current_size > 16) {
      std::this_thread::sleep_for(std::chrono::microseconds(current_size - 16));
    }
  }

  void arrival_queue_push_back(std::shared_ptr<std::vector<char>> b, int from) {
    size_t current_size = 0;
    {
      std::scoped_lock lock(m_arrival_queue_mutex);
      m_arrival_queue.push_back(std::make_pair(b, from));
      current_size = m_arrival_queue.size();
    }
    if (current_size > 16) {
      std::this_thread::sleep_for(std::chrono::microseconds(current_size - 16));
    }
  }

  // Used if dest = m_comm_rank
  template <typename Lambda, typename... Args>
  int32_t local_receive(Lambda l, const Args &...args) {
    ASSERT_DEBUG(sizeof(Lambda) == 1);
    // Question: should this be std::forward(...)
    // \pp was: (l)(this, m_comm_rank, args...);
    ygm::meta::apply_optional(l, std::make_tuple(this, m_comm_rank),
                              std::make_tuple(args...));
    return 1;
  }

  template <typename Lambda, typename... PackArgs>
  std::vector<char> pack_lambda(Lambda l, const PackArgs &...args) {
    std::vector<char> to_return;
    const std::tuple<PackArgs...> tuple_args(
        std::forward<const PackArgs>(args)...);
    ASSERT_DEBUG(sizeof(Lambda) == 1);

    void (*fun_ptr)(impl *, int, cereal::YGMInputArchive &) =
        [](impl *t, int from, cereal::YGMInputArchive &bia) {
          std::tuple<PackArgs...> ta;
          bia(ta);
          Lambda *pl;
          auto t1 = std::make_tuple((impl *)t, from);

          // \pp was: std::apply(*pl, std::tuple_cat(t1, ta));
          ygm::meta::apply_optional(*pl, std::move(t1), std::move(ta));
        };

    cereal::YGMOutputArchive oarchive(to_return); // Create an output archive
                                                  // // oarchive(fun_ptr);
    int64_t iptr = (int64_t)fun_ptr - (int64_t)&reference;
    oarchive(iptr, tuple_args);

    return to_return;
  }

  // this is used to fix address space randomization
  static void reference() {}

  bool transit_queue_process() {
    bool received = false;
    while (true) {
      auto buffer_source = transit_queue_try_pop();
      auto buffer = buffer_source.first;
      if (buffer == nullptr) {
        break;
      }
      int from = buffer_source.second;
      received = true;
      char *bitr = &buffer->at(0);
      int step = 0;

      // Read each header and step to next header based on each size until null
      // terminator
      while (*bitr != '\0') {
        ASSERT_DEBUG(step < buffer->size());

        char *hdr_stream = &(*bitr);
        header_t *hdr = reinterpret_cast<header_t *>(hdr_stream);
        char *begin_pack = &(*bitr);
        char *next_pack =
            begin_pack + hdr->len + sizeof(header_t); // pack = header + data
        if (hdr->len + sizeof(header_t) + final_send_buffers[hdr->dst]->size() >
            m_buffer_capacity - sizeof(char)) {
          async_final_flush(hdr->dst);
        }

        final_send_buffers[hdr->dst]->insert(
            final_send_buffers[hdr->dst]->end(), begin_pack, (next_pack));

        step += hdr->len + sizeof(header_t);
        bitr = &buffer->at(step);
        if (hdr->dst == m_comm_remote_rank) {
          m_self_buffer_count = step;
        }
      }
      // Only keep buffers of size m_buffer_capacity in pool of buffers
      if (buffer->size() == m_buffer_capacity)
        free_buffer(buffer);
    }
    return received;
  }

  bool arrival_queue_process() {
    bool received = false;
    while (true) {
      auto buffer_source = arrival_queue_try_pop();
      auto buffer = buffer_source.first;
      if (buffer == nullptr) {
        break;
      }
      int from = buffer_source.second;
      received = true;
      char *bitr = &buffer->at(0);
      int buffer_size = buffer->size();
      int step = 0;

      // Process large messages the original way while small messages processed
      // header to header
      if (from >= 0) {
        process_large_buffer(bitr, from, buffer_size);
      } else {
        while (*bitr != '\0') {
          ASSERT_DEBUG(step < buffer->size());

          char *hdr_stream = &(*bitr);
          header_t *hdr = reinterpret_cast<header_t *>(hdr_stream);
          // std::cout<<"\n"<<rank()<<" is taking step  : "<<step<<" len:
          // "<<hdr->len<<"\n";

          char *begin_pack = &(*bitr);
          char *begin_msg = &(*bitr) + sizeof(header_t);
          char *next_pack =
              begin_pack + hdr->len + sizeof(header_t); // pack = header + msg
          cereal::YGMInputArchive iarchive(
              begin_msg, hdr->len); // Seems pretty inefficient to deserialize
                                    // every message. Needs a better way
          int64_t iptr;
          iarchive(iptr);

          iptr += (int64_t)&reference;
          void (*fun_ptr)(impl *, int, cereal::YGMInputArchive &);
          memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
          fun_ptr(this, hdr->src, iarchive);

          step += hdr->len + sizeof(header_t);
          bitr = &buffer->at(step);
          m_recv_count++;
          // break;
        }
      }
      // Only keep buffers of size m_buffer_capacity in pool of buffers
      if (buffer->size() == m_buffer_capacity)
        free_buffer(buffer);
    }
    return received;
  }

  void process_large_buffer(char *bitr, int from, int buffer_size) {
    cereal::YGMInputArchive iarchive(bitr, buffer_size);
    while (!iarchive.empty()) {
      int64_t iptr;
      iarchive(iptr);
      iptr += (int64_t)&reference;
      void (*fun_ptr)(impl *, int, cereal::YGMInputArchive &);
      memcpy(&fun_ptr, &iptr, sizeof(uint64_t));
      fun_ptr(this, from, iarchive);
    }
    m_recv_count++;
  }

  inline void init_local_remote_comms() {
    // Local indices
    ASSERT_MPI(MPI_Comm_split_type(m_comm_async, MPI_COMM_TYPE_SHARED,
                                   m_comm_rank, MPI_INFO_NULL, &m_comm_local));
    ASSERT_MPI(MPI_Comm_size(m_comm_local, &m_comm_local_size));
    ASSERT_MPI(MPI_Comm_rank(m_comm_local, &m_comm_local_rank));

    // remote indices
    ASSERT_MPI(MPI_Comm_split(m_comm_async, m_comm_local_rank, m_comm_rank,
                              &m_comm_remote));
    ASSERT_MPI(MPI_Comm_size(m_comm_remote, &m_comm_remote_size));
    ASSERT_MPI(MPI_Comm_rank(m_comm_remote, &m_comm_remote_rank));
  }

  MPI_Comm m_comm_async;
  MPI_Comm m_comm_large_async;
  MPI_Comm m_comm_barrier;
  MPI_Comm m_comm_other;
  MPI_Comm m_comm_local;
  MPI_Comm m_comm_remote;
  int m_comm_size;
  int m_comm_rank;
  int m_comm_local_size;
  int m_comm_local_rank;
  int m_comm_remote_size;
  int m_comm_remote_rank;
  size_t m_buffer_capacity;
  int m_self_buffer_count;

  std::vector<std::shared_ptr<std::vector<char>>> m_vec_send_buffers;
  std::vector<std::shared_ptr<std::vector<char>>> intermediate_send_buffers;
  std::vector<std::shared_ptr<std::vector<char>>> final_send_buffers;
  std::mutex m_vec_free_buffers_mutex;
  std::vector<std::shared_ptr<std::vector<char>>> m_vec_free_buffers;

  std::deque<std::pair<std::shared_ptr<std::vector<char>>, int>>
      m_receive_queue;
  std::deque<std::pair<std::shared_ptr<std::vector<char>>, int>>
      m_transit_queue;
  std::deque<std::pair<std::shared_ptr<std::vector<char>>, int>>
      m_arrival_queue;
  std::mutex m_receive_queue_mutex;
  std::mutex m_transit_queue_mutex;
  std::mutex m_arrival_queue_mutex;

  std::thread m_large_listener;
  // std::thread m_small_listener;
  std::thread m_local_listener;
  std::thread m_remote_listener;

  int64_t m_recv_count = 0;
  int64_t m_send_count = 0;
  int64_t inter_send_count = 0;
  int64_t inter_recv_count = 0;

  int64_t m_local_rpc_calls = 0;
  int64_t m_local_bytes_sent = 0;

  int large_message_announce_tag = 32766;
  int large_message_tag = 32767;
};

inline comm::comm(int *argc, char ***argv,
                  int buffer_capacity = test_buffer_capacity) {
  pimpl_if = std::make_shared<detail::mpi_init_finalize>(argc, argv);
  pimpl = std::make_shared<comm::impl>(MPI_COMM_WORLD, buffer_capacity);
}

inline comm::comm(MPI_Comm mcomm, int buffer_capacity = test_buffer_capacity) {
  pimpl_if.reset();
  int flag(0);
  ASSERT_MPI(MPI_Initialized(&flag));
  if (!flag) {
    throw std::runtime_error("ERROR: MPI not initialized");
  }
  int provided(0);
  ASSERT_MPI(MPI_Query_thread(&provided));
  if (provided != MPI_THREAD_MULTIPLE) {
    throw std::runtime_error("ERROR: MPI_THREAD_MULTIPLE not provided");
  }
  pimpl = std::make_shared<comm::impl>(mcomm, buffer_capacity);
}

inline comm::~comm() {
  ASSERT_RELEASE(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  pimpl.reset();
  ASSERT_RELEASE(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  pimpl_if.reset();
}

template <typename AsyncFunction, typename... SendArgs>
inline void comm::async(int dest, AsyncFunction fn, const SendArgs &...args) {
  static_assert(std::is_empty<AsyncFunction>::value,
                "Only stateless lambdas are supported");
  pimpl->async(dest, fn, std::forward<const SendArgs>(args)...);
}

inline int comm::size() const { return pimpl->size(); }
inline int comm::rank() const { return pimpl->rank(); }
inline int comm::local_size() const { return pimpl->local_size(); }
inline int comm::local_rank() const { return pimpl->local_rank(); }

inline int64_t comm::local_bytes_sent() const {
  return pimpl->local_bytes_sent();
}

inline int64_t comm::global_bytes_sent() const {
  return all_reduce_sum(local_bytes_sent());
}

inline void comm::reset_bytes_sent_counter() {
  pimpl->reset_bytes_sent_counter();
}

inline int64_t comm::local_rpc_calls() const {
  return pimpl->local_rpc_calls();
}

inline int64_t comm::global_rpc_calls() const {
  return all_reduce_sum(local_rpc_calls());
}

inline void comm::reset_rpc_call_counter() { pimpl->reset_rpc_call_counter(); }

inline void comm::barrier() { pimpl->barrier(); }

inline void comm::async_flush(int rank) { pimpl->async_flush(rank); }

inline void comm::async_flush_all() { pimpl->async_flush_all(); }

template <typename T> inline T comm::all_reduce_sum(const T &t) const {
  return pimpl->all_reduce_sum(t);
}

template <typename T> inline T comm::all_reduce_min(const T &t) const {
  return pimpl->all_reduce_min(t);
}

template <typename T> inline T comm::all_reduce_max(const T &t) const {
  return pimpl->all_reduce_max(t);
}

template <typename T, typename MergeFunction>
inline T comm::all_reduce(const T &t, MergeFunction merge) {
  return pimpl->all_reduce(t, merge);
}

} // namespace ygm