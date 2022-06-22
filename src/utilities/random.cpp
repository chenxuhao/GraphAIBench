#include "random.h"
#include <iostream>
#include <boost/thread.hpp>

// random seeding
int64_t cluster_seedgen() {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }
  std::cout << "System entropy source not available, "
               "using fallback algorithm to generate seed instead.";
  if (f) fclose(f);
  pid  = getpid();
  s    = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}

// Make sure each thread can have different values.
static boost::thread_specific_ptr<Context> thread_instance_;

Context& Context::Get() {
  if (!thread_instance_.get()) {
    thread_instance_.reset(new Context());
  }
  return *(thread_instance_.get());
}

Context::Context() : random_generator_() {}

Context::~Context() { }

void Context::set_random_seed(const unsigned int seed) {
  Get().random_generator_.reset(new RNG(seed));
}

class Context::RNG::Generator {
 public:
  Generator() : rng_(new rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new rng_t(seed)) {}
  rng_t* rng() { return rng_.get(); }
 private:
  shared_ptr<rng_t> rng_;
};

Context::RNG::RNG() : generator_(new Generator()) { }

Context::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

Context::RNG& Context::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Context::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#ifdef ENABLE_GPU
#include "gpu_context.h"
gpu_context gpu_ctx;
cublasHandle_t gpu_context::cublas_handle_        = 0;
cusparseHandle_t gpu_context::cusparse_handle_    = 0;
cusparseMatDescr_t gpu_context::cusparse_matdescr_ = 0;
curandGenerator_t gpu_context::curand_generator_   = 0;

gpu_context::gpu_context() {
  CUBLAS_CHECK(cublasCreate(&cublas_handle_));
  CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
  CUSPARSE_CHECK(cusparseCreateMatDescr(&cusparse_matdescr_));
  CUSPARSE_CHECK(cusparseSetMatType(cusparse_matdescr_, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(cusparse_matdescr_, CUSPARSE_INDEX_BASE_ZERO));
  CURAND_CHECK(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));
  //CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen()));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator_, 1));
}
#endif
