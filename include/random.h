// random number generators for CPU
#pragma once
#include <omp.h>
#include <random>
#include <iterator>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/bernoulli_distribution.hpp>
using boost::shared_ptr;
typedef boost::mt19937 rng_t;

class Context {
 public:
  ~Context();
  static Context& Get();
  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };  

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }   
    return *(Get().random_generator_);
  }
  static void set_random_seed(const unsigned int seed);
 protected:
  shared_ptr<RNG> random_generator_;
 private:
  Context();
};

inline rng_t* rng() {
  return static_cast<rng_t*>(Context::rng_stream().generator());
}

class PerThreadRNG {
  std::vector<std::default_random_engine> engines;
  std::vector<std::uniform_real_distribution<float>> distributions;
public:
  PerThreadRNG() {
    auto nt = omp_get_num_threads();
    engines.resize(nt);
    for (int tid = 0; tid < nt; tid++) {
      engines[tid].seed(tid*nt);
      std::uniform_real_distribution<float> dst(0.0, 0.1);
      distributions.push_back(dst);
    }
  }
  float get_number() {
    auto tid = omp_get_thread_num();
    float num = distributions[tid](engines[tid]);
    return num;
  }
};

class random_generator {
public:
  static random_generator& get_instance() {
    static random_generator instance;
    return instance;
  }
  std::mt19937& operator()() { return gen_; }
  void set_seed(unsigned int seed) { gen_.seed(seed); }

private:
  random_generator() : gen_(1) {}
  std::mt19937 gen_;
};

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_int_distribution<T> dst(min, max);
  return dst(random_generator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
uniform_rand(T min, T max) {
  std::uniform_real_distribution<T> dst(min, max);
  return dst(random_generator::get_instance()());
}

