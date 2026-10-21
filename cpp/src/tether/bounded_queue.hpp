// A first-in, first-out queue of at most `capacity` items, safe between threads:
// the link between two parallel pipeline stages (docs/CONCEPTS.md, "Bounded
// queues and backpressure"). A full queue makes the stage before it wait.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace tether {

template <class T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

  // Waits while the queue is full. False, and `item` dropped, once closed.
  bool push(T item) {
    std::unique_lock lock(mutex_);
    not_full_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
    if (closed_) return false;
    items_.push_back(std::move(item));
    not_empty_.notify_one();
    return true;
  }

  // Waits while the queue is empty and open. Once closed, the items still held
  // come out in order, then nullopt.
  std::optional<T> pop() {
    std::unique_lock lock(mutex_);
    not_empty_.wait(lock, [&] { return closed_ || !items_.empty(); });
    if (items_.empty()) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    not_full_.notify_one();
    return item;
  }

  // No more pushes; wakes everyone waiting. Safe to call more than once.
  void close() {
    std::lock_guard lock(mutex_);
    closed_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
  }

 private:
  size_t capacity_;
  std::mutex mutex_;
  std::condition_variable not_full_, not_empty_;
  std::deque<T> items_;
  bool closed_ = false;
};

}  // namespace tether
