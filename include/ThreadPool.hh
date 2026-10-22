#pragma once

#include <vector>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>  
#include <stdexcept>

namespace QTensorNet
{    
    class ThreadPool
    {
        private:
            struct ActiveGuard
            {
                std::atomic<size_t>& counter_;
            
                explicit ActiveGuard(std::atomic<size_t>& counter) noexcept : counter_(counter)
                {
                
                }
            
                ActiveGuard(const ActiveGuard&) = delete;
                ActiveGuard& operator=(const ActiveGuard&) = delete;
            
                ~ActiveGuard() noexcept
                {
                    counter_.fetch_sub(1UL, std::memory_order_relaxed);
                }
            };
        
        public:
            ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
            {
                if(num_threads == 0)
                {
                    num_threads = 1;
                }
                
                try
                {
                    for(size_t i = 0; i < num_threads; ++i)
                    {
                        threads_.emplace_back([this]()
                        {
                            while(true)
                            {
                                std::function<void()> task;
                            
                                {
                                    std::unique_lock<std::mutex> lock(mutex_);
                                
                                    cv_.wait(lock, [this](){return stop_ || !tasks_.empty();});
                                
                                    if(stop_ && tasks_.empty())
                                    {
                                        return;
                                    }
                                
                                    task = std::move(tasks_.front());
                                    tasks_.pop();
                                }

                                active_.fetch_add(1UL, std::memory_order_relaxed);

                                {
                                    ActiveGuard guard{active_};
                                
                                    try
                                    {
                                        task();
                                    }
                                    catch(...){}
                                }
                            }
                        });
                    }
                }
                catch(...)
                {
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        
                        stop_ = true;
                    }
                
                    cv_.notify_all();
                
                    for(auto& t : threads_) 
                    {
                        if(t.joinable())
                        {
                            t.join();
                        }
                    }
                
                    throw;
                }
            }
        
            ~ThreadPool()
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);

                    stop_ = true;
                }
            
                cv_.notify_all();
            
                for(auto& thread : threads_)
                {
                    if(thread.joinable())
                    {
                        thread.join();
                    }
                }
            }
        
            ThreadPool(const ThreadPool&) = delete;
            ThreadPool& operator=(const ThreadPool&) = delete;
        
            template<typename F, typename... Args>
            auto AddTask(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
            {
                using returnType = std::invoke_result_t<F, Args...>;
            
                auto task = std::make_shared<std::packaged_task<returnType()>>(
                    [f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable -> returnType
                    {
                        return std::invoke(std::move(f), std::forward<decltype(args)>(args)...);
                    });
                
                std::future<returnType> future = task->get_future();
                
                {
                    std::lock_guard<std::mutex> lock(mutex_);

                    if(stop_)
                    {
                        throw std::runtime_error("ThreadPool: AddTask called after shutdown");
                    }

                    tasks_.emplace([task]() {(*task)();});
                }
            
                cv_.notify_one();
            
                return future;
            }
        
            size_t GetSize() const noexcept
            {
                return threads_.size();
            }
        
            size_t GetActiveTasks() const noexcept
            {
                return active_.load(std::memory_order_relaxed);
            }

            size_t GetQueuedTasks() const
            {
                std::lock_guard<std::mutex> lock(mutex_);
            
                return tasks_.size();
            }
        
        private:
            std::vector<std::thread> threads_;
            std::queue<std::function<void()>> tasks_;
            mutable std::mutex mutex_;
            std::condition_variable cv_;
            std::atomic<size_t> active_{0UL};
            bool stop_{false};
    };
}