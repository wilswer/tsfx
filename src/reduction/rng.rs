//! A small seeded generator whose stream never changes with a dependency
//! update, so a seed gives the same row sample and shadow features in every
//! tsfx version.

/// SplitMix64 (Steele, Lea and Flood, 2014).
#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform integer in `0..n`, by rejection (no modulo bias).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        let n = n as u64;
        // The largest multiple of n that fits in a u64: draws at or above it
        // would favor the small values
        let limit = u64::MAX - u64::MAX % n;
        loop {
            let x = self.next_u64();
            if x < limit {
                return (x % n) as usize;
            }
        }
    }

    /// Shuffles `xs` in place (Fisher-Yates, from the end).
    pub(crate) fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }

    /// `k` distinct indices of `0..n`, in ascending order.
    pub(crate) fn sample_sorted(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample_sorted({n}, {k})");
        let mut idx: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            idx.swap(i, j);
        }
        idx.truncate(k);
        idx.sort_unstable();
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn test_splitmix64_reference() {
        // SplitMix64 reference stream for seed 0
        let mut g = SplitMix64::new(0);
        assert_eq!(g.next_u64(), 0xe220a8397b1dcdaf);
    }

    #[test]
    fn test_sample_sorted() {
        let s = SplitMix64::new(1).sample_sorted(100, 10);
        assert_eq!(s.len(), 10);
        assert!(s.windows(2).all(|w| w[0] < w[1]) && *s.last().unwrap() < 100);
        assert_eq!(s, SplitMix64::new(1).sample_sorted(100, 10));
        assert_eq!(SplitMix64::new(2).sample_sorted(5, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_shuffle_is_a_permutation() {
        let mut xs: Vec<usize> = (0..50).collect();
        SplitMix64::new(3).shuffle(&mut xs);
        assert_ne!(xs, (0..50).collect::<Vec<_>>());
        xs.sort_unstable();
        assert_eq!(xs, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn test_below_stays_in_range() {
        let mut g = SplitMix64::new(4);
        assert!((0..1000).all(|_| g.below(7) < 7));
        assert_eq!(g.below(1), 0);
    }
}
