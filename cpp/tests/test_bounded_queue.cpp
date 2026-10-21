// The bounded queue between parallel pipeline stages.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "tether/bounded_queue.hpp"

using namespace tether;
using namespace std::chrono_literals;

TEST(BoundedQueue, GivesItemsBackInTheOrderTheyWentIn) {
  BoundedQueue<int> q(3);
  ASSERT_TRUE(q.push(1));
  ASSERT_TRUE(q.push(2));
  ASSERT_TRUE(q.push(3));
  EXPECT_EQ(q.pop(), 1);
  EXPECT_EQ(q.pop(), 2);
  EXPECT_EQ(q.pop(), 3);
}

TEST(BoundedQueue, PushWaitsWhileFullUntilAPopMakesRoom) {
  BoundedQueue<int> q(1);
  ASSERT_TRUE(q.push(1));
  std::atomic<bool> pushed = false;
  std::thread producer([&] {
    q.push(2);
    pushed = true;
  });
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(pushed);  // still blocked: the queue holds its one item
  EXPECT_EQ(q.pop(), 1);
  producer.join();
  EXPECT_TRUE(pushed);
  EXPECT_EQ(q.pop(), 2);
}

TEST(BoundedQueue, PopWaitsWhileEmptyUntilAPush) {
  BoundedQueue<int> q(1);
  std::optional<int> got;
  std::thread consumer([&] { got = q.pop(); });
  std::this_thread::sleep_for(50ms);
  EXPECT_FALSE(got);
  q.push(7);
  consumer.join();
  EXPECT_EQ(got, 7);
}

TEST(BoundedQueue, AfterCloseItDrainsWhatItHoldsThenSaysItIsDone) {
  BoundedQueue<int> q(2);
  q.push(1);
  q.push(2);
  q.close();
  EXPECT_EQ(q.pop(), 1);
  EXPECT_EQ(q.pop(), 2);
  EXPECT_EQ(q.pop(), std::nullopt);
  EXPECT_EQ(q.pop(), std::nullopt);
}

TEST(BoundedQueue, PushFailsOnceClosed) {
  BoundedQueue<int> q(2);
  q.close();
  EXPECT_FALSE(q.push(1));
  EXPECT_EQ(q.pop(), std::nullopt);
}

TEST(BoundedQueue, CloseWakesAPushBlockedOnAFullQueue) {
  BoundedQueue<int> q(1);
  q.push(1);
  std::atomic<int> result = -1;
  std::thread producer([&] { result = q.push(2) ? 1 : 0; });
  std::this_thread::sleep_for(50ms);
  q.close();
  producer.join();
  EXPECT_EQ(result, 0);
}

TEST(BoundedQueue, CloseWakesAPopBlockedOnAnEmptyQueue) {
  BoundedQueue<int> q(1);
  std::optional<int> got = 0;
  std::thread consumer([&] { got = q.pop(); });
  std::this_thread::sleep_for(50ms);
  q.close();
  consumer.join();
  EXPECT_EQ(got, std::nullopt);
}

TEST(BoundedQueue, OneProducerOneConsumerKeepEveryItemInOrder) {
  BoundedQueue<int> q(2);
  std::thread producer([&] {
    for (int i = 0; i < 1000; ++i) q.push(i);
    q.close();
  });
  std::vector<int> got;
  while (auto item = q.pop()) got.push_back(*item);
  producer.join();
  ASSERT_EQ(got.size(), 1000u);
  for (int i = 0; i < 1000; ++i) ASSERT_EQ(got[i], i);
}
