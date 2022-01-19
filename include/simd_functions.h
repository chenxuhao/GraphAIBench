#include <immintrin.h>
//#include <vector>
//#include <assert.h>

// CPUs support RAM access like this: "ymmword ptr [rax+64]"
// Using templates with offset int argument to make easier for compiler to emit good code.

// Multiply 8 floats by another 8 floats.
template<int offsetRegs>
inline __m256 mul8(const float* p1, const float* p2) {
  constexpr int lanes = offsetRegs * 8;
  const __m256 a = _mm256_loadu_ps( p1 + lanes );
  const __m256 b = _mm256_loadu_ps( p2 + lanes );
  return _mm256_mul_ps( a, b );
}

// Returns acc + ( p1 * p2 ), for 8-wide float lanes.
template<int offsetRegs>
inline __m256 fma8( __m256 acc, const float* p1, const float* p2) {
  constexpr int lanes = offsetRegs * 8;
  const __m256 a = _mm256_loadu_ps( p1 + lanes );
  const __m256 b = _mm256_loadu_ps( p2 + lanes );
  return _mm256_fmadd_ps( a, b, acc );
}

// Multiply 16 floats by another 16 floats.
template<int offsetRegs>
inline __m512 mul16(const float* p1, const float* p2) {
  constexpr int lanes = offsetRegs * 16;
  const __m512 a = _mm512_loadu_ps( p1 + lanes );
  const __m512 b = _mm512_loadu_ps( p2 + lanes );
  return _mm512_mul_ps( a, b );
}

// Returns acc + ( p1 * p2 ), for 16-wide float lanes.
template<int offsetRegs>
inline __m512 fma16(__m512 acc, const float* p1, const float* p2) {
  constexpr int lanes = offsetRegs * 16;
  const __m512 a = _mm512_loadu_ps( p1 + lanes );
  const __m512 b = _mm512_loadu_ps( p2 + lanes );
  return _mm512_fmadd_ps( a, b, acc );
}

inline float reduce16(__m512 r16) {
  // Add 16 values into 8
  const __m256 r8 = _mm256_add_ps(_mm512_castps512_ps256(r16), _mm512_extractf32x8_ps(r16, 1));
  // Add 8 values into 4
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  // Add 4 values into 2
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  // Add 2 lower values into the final result
  const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
  // Return the lowest lane of the result vector.
  // The intrinsic below compiles into noop, modern compilers return floats in the lowest lane of xmm0 register.
  return _mm_cvtss_f32(r1);
}

inline float reduce8(__m256 r8) {
  const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
  const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
  const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
  return _mm_cvtss_f32(r1);
}

