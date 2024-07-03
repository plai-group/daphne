(ns daphne.linalg)


(def foppl-linalg
  '[(defn const-vector [value size]
      (vec (repeat size value)))

    (defn elem-add [tensor elem]
      (if (vector? (first tensor))
        (foreach (count tensor) [row tensor]
                 (elem-add row elem))
        (foreach (count tensor) [v tensor] (+ v elem))))

    (defn dot-helper [t state a b]
      (+ state
         (* (get a t)
            (get b t))))

    (defn dot [a b]
      (loop (count a) 0 dot-helper a b))

    (defn row-mul [t state m v]
      (conj state (dot (get m t) v)))

    (defn mmul [m v]
      (loop (count m) [] row-mul m v))

    (defn row-helper [i sum a b]
      (+ sum
         (dot (get a i)
              (get b i))))

    (defn inner-square [a b]
      (loop (count a) 0 row-helper a b))

    (defn inner-cubic [a b]
      (apply + (foreach (count a) [n (range (count a))]
                        (inner-square (get a n) (get b n)))))

    (defn slice-square [input size stride i j]
      (foreach size [k (range (* i stride)
                              (+ size (* i stride)))]
               (subvec (get input k)
                       (* j stride)
                       (+ size (* j stride)))))

    (defn slice-cubic [inputs size stride i j]
      (foreach (count inputs) [input inputs]
               (slice-square input size stride i j)))

    (defn conv-helper [inputs kernel bias stride]
      (let [ic (count (first inputs))
            size (count (first kernel))
            remainder (- size stride)
            to-cover (- ic remainder)
            iters (int (Math/floor (/ to-cover stride)))]
        (foreach iters [i (range iters)]
                 (foreach iters [j (range iters)]
                          (inner-cubic (slice-cubic inputs size stride i j)
                                       kernel)))))

    (defn conv2d [inputs kernels bias stride]
      (foreach (count kernels) [ksi (range (count kernels))]
               (conv-helper inputs (get kernels ksi) (get bias ksi) stride)))


    (defn sample-layer [hidden-layer]
      (foreach (count hidden-layer) [hi hidden-layer]
               (foreach (count hi) [hii hi]
                        (foreach (count hii) [hiii hii]
                                 (sample (normal hiii 1) 0)))))])
