/**
 *  @brief SVE backend for substring & byte-set search.
 *  @file include/stringzilla/find/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SVE_H_
#define STRINGZILLA_FIND_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#if SZ_USE_SVE
/**
 *  @brief How far from the start (or, backwards, the end) of the haystack the substring search keeps the
 *         low-latency loop before switching to the high-throughput one. Matches of frequent needles fall within
 *         it, and on long scans of rare needles it is negligible. Graviton 4 and 5 measurements.
 */
#define SZ_FIND_SVE_NEAR_BYTES 512
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

/**
 *  @brief XOR of the haystack bytes at up to three probe offsets with the needle bytes there, OR-ed together:
 *         a lane is zero exactly where all probes match. Staying in the vector domain keeps a single
 *         predicate-producing compare per register, instead of one per probe plus predicate ANDs, which
 *         is what limits the predicate-heavy loop on Neoverse V1 (Graviton 3).
 */
SZ_HELPER_INLINE svuint8_t sz_find_probes_diff_sve_(svbool_t pred, sz_u8_t const *text, sz_size_t const offsets[3],
                                                    svuint8_t first, svuint8_t second, svuint8_t third, int probes) {
    svuint8_t diff = sveor_u8_x(pred, svld1_u8(pred, text + offsets[0]), first);
    if (probes > 1) diff = svorr_u8_x(pred, diff, sveor_u8_x(pred, svld1_u8(pred, text + offsets[1]), second));
    if (probes > 2) diff = svorr_u8_x(pred, diff, sveor_u8_x(pred, svld1_u8(pred, text + offsets[2]), third));
    return diff;
}

/** Lanes of the block at @p text where all probes match. One probe compares directly. */
SZ_HELPER_INLINE svbool_t sz_find_probes_match_sve_(svbool_t pred, sz_u8_t const *text, sz_size_t const offsets[3],
                                                    svuint8_t first, svuint8_t second, svuint8_t third, int probes) {
    if (probes == 1) return svcmpeq_u8(pred, svld1_u8(pred, text + offsets[0]), first);
    return svcmpeq_n_u8(pred, sz_find_probes_diff_sve_(pred, text, offsets, first, second, third, probes), 0);
}

/**
 *  @brief The first candidate of @p matches in the block at @p block that holds the whole needle, or NULL.
 *         With @p verify false the probes cover the needle and @p matches must not be empty.
 */
SZ_HELPER_INLINE sz_cptr_t sz_find_probes_resolve_sve_(svbool_t pred, svbool_t matches, sz_cptr_t block,
                                                       sz_cptr_t needle, sz_size_t needle_length, int verify) {
    if (!verify) return block + svcntp_b8(pred, svbrkb_b_z(pred, matches));
    while (svptest_any(pred, matches)) {
        svbool_t const before = svbrkb_b_z(pred, matches);
        sz_cptr_t const candidate = block + svcntp_b8(pred, before);
        if (sz_equal_sve(candidate, needle, needle_length)) return candidate;
        // Clear the first match and continue.
        matches = svbic_b_z(pred, matches, svpnext_b8(pred, before));
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Forward search comparing up to three probe bytes of the needle in every candidate position,
 *         verifying the whole needle on probe matches when @p verify is set.
 *
 *  One predicated register goes first, as most matches are close and many haystacks are short.
 *  Then two full registers per iteration, testing both at once with a single compare of their minimum
 *  against zero. A predicated register at a time covers the rest.
 */
SZ_HELPER_INLINE sz_cptr_t sz_find_probes_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length, sz_size_t const offsets[3], int probes,
                                               int verify) {
    sz_u8_t const *const text = (sz_u8_t const *)haystack;
    svuint8_t const first = svdup_n_u8(((sz_u8_t const *)needle)[offsets[0]]);
    svuint8_t const second = svdup_n_u8(((sz_u8_t const *)needle)[offsets[1]]);
    svuint8_t const third = svdup_n_u8(((sz_u8_t const *)needle)[offsets[2]]);
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const candidates = haystack_length - needle_length + 1;
    svbool_t const all = svptrue_b8();
    sz_cptr_t found;

    svbool_t pred = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)candidates);
    svbool_t matches = sz_find_probes_match_sve_(pred, text, offsets, first, second, third, probes);
    if (svptest_any(pred, matches) &&
        (!verify || (found = sz_find_probes_resolve_sve_(pred, matches, haystack, needle, needle_length, verify))))
        return verify ? found : haystack + svcntp_b8(pred, svbrkb_b_z(pred, matches));
    if (candidates <= vector_bytes) return SZ_NULL_CHAR;

    sz_size_t progress = vector_bytes;
    // Near the start, two full registers per iteration, each tested with its own flag-setting compare. The
    // register holding a match is known without extra work, which keeps the latency from a match to the result
    // short: for frequent needles that latency, not the scan throughput, bounds the next search.
    sz_size_t const near_end = candidates < SZ_FIND_SVE_NEAR_BYTES ? candidates : SZ_FIND_SVE_NEAR_BYTES;
    for (; progress + 2 * vector_bytes <= near_end; progress += 2 * vector_bytes) {
        svbool_t const matches_first = svcmpeq_n_u8(
            all, sz_find_probes_diff_sve_(all, text + progress, offsets, first, second, third, probes), 0);
        svbool_t const matches_second = svcmpeq_n_u8(
            all, sz_find_probes_diff_sve_(all, text + progress + vector_bytes, offsets, first, second, third, probes),
            0);
        if (svptest_any(all, matches_first)) {
            if (!verify) return haystack + progress + svcntp_b8(all, svbrkb_b_z(all, matches_first));
            if ((found = sz_find_probes_resolve_sve_(all, matches_first, haystack + progress, needle, needle_length,
                                                     verify)))
                return found;
        }
        if (svptest_any(all, matches_second)) {
            if (!verify) return haystack + progress + vector_bytes + svcntp_b8(all, svbrkb_b_z(all, matches_second));
            if ((found = sz_find_probes_resolve_sve_(all, matches_second, haystack + progress + vector_bytes, needle,
                                                     needle_length, verify)))
                return found;
        }
    }

    // Further on, throughput matters more: one compare of the minimum of both registers against zero.
    for (; progress + 2 * vector_bytes <= candidates; progress += 2 * vector_bytes) {
        svuint8_t const diff_first = sz_find_probes_diff_sve_(all, text + progress, offsets, first, second, third, probes);
        svuint8_t const diff_second = sz_find_probes_diff_sve_(all, text + progress + vector_bytes, offsets, first,
                                                               second, third, probes);
        if (!svptest_any(all, svcmpeq_n_u8(all, svmin_u8_x(all, diff_first, diff_second), 0))) continue;
        // Pick the first candidate of the two registers without a branch on which of them holds it.
        svbool_t const matches_first = svcmpeq_n_u8(all, diff_first, 0);
        svbool_t const matches_second = svcmpeq_n_u8(all, diff_second, 0);
        sz_size_t const skip_first = svcntp_b8(all, svbrkb_b_z(all, matches_first));
        sz_size_t const skip_second = svcntp_b8(all, svbrkb_b_z(all, matches_second));
        sz_cptr_t const candidate = haystack + progress +
                                    (skip_first < vector_bytes ? skip_first : vector_bytes + skip_second);
        if (!verify || sz_equal_sve(candidate, needle, needle_length)) return candidate;
        if ((found = sz_find_probes_resolve_sve_(all, matches_first, haystack + progress, needle, needle_length,
                                                 verify)))
            return found;
        if ((found = sz_find_probes_resolve_sve_(all, matches_second, haystack + progress + vector_bytes, needle,
                                                 needle_length, verify)))
            return found;
    }

    // Fewer than two registers remain: two predicated steps without a loop, so no trip count to mispredict.
    for (int step = 0; step != 2; ++step, progress += vector_bytes) {
        pred = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)candidates);
        matches = sz_find_probes_match_sve_(pred, text + progress, offsets, first, second, third, probes);
        if (svptest_any(pred, matches) &&
            (!verify ||
             (found = sz_find_probes_resolve_sve_(pred, matches, haystack + progress, needle, needle_length, verify))))
            return verify ? found : haystack + progress + svcntp_b8(pred, svbrkb_b_z(pred, matches));
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief The last candidate of @p matches in the block at @p block that holds the whole needle, or NULL.
 *         With @p verify false the probes cover the needle and @p matches must not be empty.
 */
SZ_HELPER_INLINE sz_cptr_t sz_rfind_probes_resolve_sve_(svbool_t matches, sz_cptr_t block, sz_cptr_t needle,
                                                        sz_size_t needle_length, int verify) {
    svbool_t const all = svptrue_b8();
    sz_size_t const last_lane = svcntb() - 1;
    if (!verify) return block + last_lane - svcntp_b8(all, svbrkb_b_z(all, svrev_b8(matches)));
    while (svptest_any(all, matches)) {
        svbool_t const after = svbrkb_b_z(all, svrev_b8(matches)); //? Lanes after the last match, reversed.
        sz_cptr_t const candidate = block + last_lane - svcntp_b8(all, after);
        if (sz_equal_sve(candidate, needle, needle_length)) return candidate;
        // Clear the last match and continue.
        matches = svbic_b_z(all, matches, svrev_b8(svpnext_b8(all, after)));
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Backward search mirroring `sz_find_probes_sve_`: one predicated register at the end of the haystack,
 *         then two full registers per iteration, then a predicated register at a time down to the start.
 */
SZ_HELPER_INLINE sz_cptr_t sz_rfind_probes_sve_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length, sz_size_t const offsets[3], int probes,
                                                int verify) {
    sz_u8_t const *const text = (sz_u8_t const *)haystack;
    svuint8_t const first = svdup_n_u8(((sz_u8_t const *)needle)[offsets[0]]);
    svuint8_t const second = svdup_n_u8(((sz_u8_t const *)needle)[offsets[1]]);
    svuint8_t const third = svdup_n_u8(((sz_u8_t const *)needle)[offsets[2]]);
    sz_size_t const vector_bytes = svcntb();
    svbool_t const all = svptrue_b8();
    sz_cptr_t found;

    // Candidates in `[0, end)` remain unchecked.
    sz_size_t end = haystack_length - needle_length + 1;
    sz_size_t start = end > vector_bytes ? end - vector_bytes : 0;
    svbool_t pred = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(end - start));
    svbool_t matches = sz_find_probes_match_sve_(pred, text + start, offsets, first, second, third, probes);
    if (svptest_any(pred, matches) &&
        (!verify || (found = sz_rfind_probes_resolve_sve_(matches, haystack + start, needle, needle_length, verify))))
        return verify ? found : sz_rfind_probes_resolve_sve_(matches, haystack + start, needle, needle_length, 0);
    if (!start) return SZ_NULL_CHAR;
    end = start;

    // Near the end, two full registers per iteration, each tested with its own flag-setting compare, then the
    // minimum of both, as in `sz_find_probes_sve_`.
    sz_size_t const near_start = end > SZ_FIND_SVE_NEAR_BYTES ? end - SZ_FIND_SVE_NEAR_BYTES : 0;
    for (; end >= near_start + 2 * vector_bytes; end -= 2 * vector_bytes) {
        svbool_t const matches_high = svcmpeq_n_u8(
            all, sz_find_probes_diff_sve_(all, text + end - vector_bytes, offsets, first, second, third, probes), 0);
        svbool_t const matches_low = svcmpeq_n_u8(
            all, sz_find_probes_diff_sve_(all, text + end - 2 * vector_bytes, offsets, first, second, third, probes),
            0);
        if (svptest_any(all, matches_high) &&
            (found = sz_rfind_probes_resolve_sve_(matches_high, haystack + end - vector_bytes, needle, needle_length,
                                                  verify)))
            return found;
        if (svptest_any(all, matches_low) &&
            (found = sz_rfind_probes_resolve_sve_(matches_low, haystack + end - 2 * vector_bytes, needle,
                                                  needle_length, verify)))
            return found;
    }
    for (; end >= 2 * vector_bytes; end -= 2 * vector_bytes) {
        svuint8_t const diff_high = sz_find_probes_diff_sve_(all, text + end - vector_bytes, offsets, first, second,
                                                             third, probes);
        svuint8_t const diff_low = sz_find_probes_diff_sve_(all, text + end - 2 * vector_bytes, offsets, first,
                                                            second, third, probes);
        if (!svptest_any(all, svcmpeq_n_u8(all, svmin_u8_x(all, diff_high, diff_low), 0))) continue;
        svbool_t const matches_high = svcmpeq_n_u8(all, diff_high, 0);
        svbool_t const matches_low = svcmpeq_n_u8(all, diff_low, 0);
        sz_size_t const skip_high = svcntp_b8(all, svbrkb_b_z(all, svrev_b8(matches_high)));
        sz_size_t const skip_low = svcntp_b8(all, svbrkb_b_z(all, svrev_b8(matches_low)));
        sz_cptr_t const candidate = haystack + end - 1 -
                                    (skip_high < vector_bytes ? skip_high : vector_bytes + skip_low);
        if (!verify || sz_equal_sve(candidate, needle, needle_length)) return candidate;
        if ((found = sz_rfind_probes_resolve_sve_(matches_high, haystack + end - vector_bytes, needle, needle_length,
                                                  verify)))
            return found;
        if ((found = sz_rfind_probes_resolve_sve_(matches_low, haystack + end - 2 * vector_bytes, needle,
                                                  needle_length, verify)))
            return found;
    }

    // Fewer than two registers remain: two predicated steps without a loop, so no trip count to mispredict.
    for (int step = 0; step != 2; ++step, end = start) {
        start = end > vector_bytes ? end - vector_bytes : 0;
        pred = svwhilelt_b8((sz_u64_t)0, (sz_u64_t)(end - start));
        matches = sz_find_probes_match_sve_(pred, text + start, offsets, first, second, third, probes);
        if (svptest_any(pred, matches) &&
            (!verify ||
             (found = sz_rfind_probes_resolve_sve_(matches, haystack + start, needle, needle_length, verify))))
            return verify ? found : sz_rfind_probes_resolve_sve_(matches, haystack + start, needle, needle_length, 0);
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_find_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_size_t const offsets[3] = {0, 0, 0};
    if (!haystack_length) return SZ_NULL_CHAR;
    return sz_find_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_size_t const offsets[3] = {0, 0, 0};
    if (!haystack_length) return SZ_NULL_CHAR;
    return sz_rfind_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
}

SZ_API_COMPTIME sz_cptr_t sz_find_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length) {
    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;

    // Needles of up to 3 bytes are probed whole and need no verification.
    if (needle_length <= 3) {
        sz_size_t const offsets[3] = {0, needle_length > 1, needle_length - 1};
        if (needle_length == 1) return sz_find_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
        if (needle_length == 2) return sz_find_probes_sve_(haystack, haystack_length, needle, 2, offsets, 2, 0);
        return sz_find_probes_sve_(haystack, haystack_length, needle, 3, offsets, 3, 0);
    }

    // For longer needles we first pick "anomalies" (i.e. informative offsets).
    sz_size_t offsets[3];
    sz_locate_needle_anomalies_(needle, needle_length, &offsets[0], &offsets[1], &offsets[2]);
    return sz_find_probes_sve_(haystack, haystack_length, needle, needle_length, offsets, 3, 1);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length) {
    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;

    // Needles of up to 3 bytes are probed whole and need no verification.
    if (needle_length <= 3) {
        sz_size_t const offsets[3] = {0, needle_length > 1, needle_length - 1};
        if (needle_length == 1) return sz_rfind_probes_sve_(haystack, haystack_length, needle, 1, offsets, 1, 0);
        if (needle_length == 2) return sz_rfind_probes_sve_(haystack, haystack_length, needle, 2, offsets, 2, 0);
        return sz_rfind_probes_sve_(haystack, haystack_length, needle, 3, offsets, 3, 0);
    }

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offsets[3];
    sz_locate_needle_anomalies_(needle, needle_length, &offsets[0], &offsets[1], &offsets[2]);
    return sz_rfind_probes_sve_(haystack, haystack_length, needle, needle_length, offsets, 3, 1);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SVE_H_
