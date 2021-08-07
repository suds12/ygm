// Copyright 2019-2021 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <ygm/comm.hpp>

int main(int argc, char **argv) {

  ygm::comm world(&argc, &argv);

  std::string small_message1 = "Ground Control to Major Tom"; // 43
  // std::string small_message2 = "T";
  std::string small_message3 = "small msg "; // 28
  std::string small_message4 = "bigggggggggggggger message ";
  std::string small_message5 = "message ";
  std::string small_message6 = "aab ";

  std::string large_message =
      "Take your protein pills and put your helmet on. Commencing countdown, "
      "engines on. Check ignition and may God's love be with you.";

  // Define the function to execute in async
  auto howdy = [](auto pcomm, int from, const std::string &str) {
    std::cout << "Howdy, I'm rank " << pcomm->rank() << " (local rank "
              << pcomm->local_rank() << ", remote rank " << pcomm->remote_rank()
              << "), and I received a message from rank " << from
              << " that read: \"" << str << "\"" << std::endl;
  };

  // 0 send small message to everyone and large message only to 1
  if (world.rank() == 2) {
    // for (int dest = 0; dest < world.size(); ++dest) {
    //   world.async(dest, howdy, small_message);
    // }

    world.async(1, howdy, small_message1);
    world.async(3, howdy, small_message3);
    world.async(4, howdy, small_message4);
    world.async(4, howdy, small_message5);
    world.async(4, howdy, small_message6);
    world.async(5, howdy, small_message6);
  }
  world.barrier();
  if (world.rank() == 4) {
    world.async(1, howdy, small_message1);
    world.async(3, howdy, small_message3);
    world.async(5, howdy, small_message4);
    world.async(2, howdy, small_message5);
    world.async(2, howdy, small_message6);
  }
  return 0;
}