(ns daphne.metropolis-within-gibbs-test
  (:require  [clojure.test :refer [deftest testing is]]
             [daphne.core :refer [sample-from-prior code->graph]]
             [anglican.runtime :refer [mean std normal sqrt
                                       discrete gamma dirichlet flip
                                       defdist]]
             [anglican.core :refer [doquery]]
             [anglican.emit :refer [query fm with-primitive-procedures]]
             [daphne.metropolis-within-gibbs :refer [metropolis-within-gibbs]]
             [daphne.test-helpers :refer [local-gensym]]
             [daphne.gensym :refer [*my-gensym*]]))


(defn tol? [foppl anglican]
  (< (/ (Math/abs (- foppl anglican))
        anglican)
     0.05))


(deftest exercise-1-test
  (testing "Test program from exercise 1"
    (let [foppl (->> (metropolis-within-gibbs '((let [mu (sample (normal 1 (sqrt 5)))
                                                      sigma (sqrt 2)
                                                      lik (normal mu sigma)]
                                                  (observe lik 8)
                                                  (observe lik 9)
                                                  mu))) 
                     (drop 10000)
                     (take 100000)) 
            anglican (->>
                      (doquery :smc 
                               (query []
                                      (let [mu (sample (normal 1 (sqrt 5)))
                                            sigma (sqrt 2)
                                            lik (normal mu sigma)]
                                        (observe lik 8)
                                        (observe lik 9)
                                        mu))
                               []
                               :number-of-particles 10000) 
                      (drop 10000)
                      (map :result)
                      (take 100000))]
      #_(prn "exercise 1:" (mean foppl) (mean anglican))
        (is (tol? (mean foppl) (mean anglican)))
        (is (tol? (std foppl) (std anglican))))))

(deftest exercise-2-test
  (testing "Testing program from exercise 2"
    (let [foppl (->> (metropolis-within-gibbs '((defn observe-data [_ data slope bias]
                                                (let [xn (first data)
                                                      yn (second data)
                                                      zn (+ (* slope xn) bias)]
                                                  (observe (normal zn 1.0) yn)
                                                  (rest (rest data))))

                                              (let [slope (sample (normal 0.0 10.0))
                                                    bias  (sample (normal 0.0 10.0))
                                                    data (vector 1.0 2.1 2.0 3.9 3.0 5.3
                                                                 4.0 7.7 5.0 10.2 6.0 12.9)]
                                                (loop 6 data observe-data slope bias)
                                                (vector slope bias))))
                     (drop 10000)
                     (take 200000))
          anglican (->>
                    (doquery :smc (let [observe-data
                                        (fm [xn yn slope bias]
                                            (let [zn (+ (* slope xn) bias)]
                                              (observe (normal zn 1.0) yn)))]

                                    (query []
                                           (let [slope (sample (normal 0.0 10.0))
                                                 bias  (sample (normal 0.0 10.0))
                                                 data (vector 1.0 2.1 2.0 3.9 3.0 5.3
                                                              4.0 7.7 5.0 10.2 6.0 12.9)
                                                 xn (take-nth 2 data)
                                                 yn (take-nth 2 (rest data))]
                                             (loop [i 0]
                                               (when (< i 6)
                                                 (observe-data (nth xn i) (nth yn i) slope bias)
                                                 (recur (inc i))))
                                             (vector slope bias))))
                             []
                             :number-of-particles 10000)
                    (drop 10000)
                    (map :result)
                    (take 200000))]

      #_(prn "exercise 2:" (mean foppl) (mean anglican))
      (is (tol? (mean (map first foppl)) (mean (map first anglican))))
      (is (tol? (mean (map second foppl)) (mean (map second anglican))))

      (is (tol? (std (map first foppl)) (std (map first anglican))))
      (is (tol? (std (map second foppl)) (std (map second anglican)))))))


(deftest exercise-3-test
  (testing "Testing program from exercise 3"
    (let [foppl (->> (metropolis-within-gibbs '((let [data [1.1 2.1 2.0 1.9 0.0 -0.1 -0.05]
                                                    likes (foreach 3 []
                                                                   (let [mu (sample (normal 0.0 10.0))
                                                                         sigma (sample (gamma 1.0 1.0))]
                                                                     (normal mu sigma)))
                                                    pi (sample (dirichlet [1.0 1.0 1.0]))
                                                    z-prior (discrete pi)
                                                    z (foreach 7 [y data]
                                                               (let [z (sample z-prior)]
                                                                 (observe (get likes z) y)
                                                                 z))]
                                                (= (first z) (second z)))))
                     (drop 20000)
                     (take 200000)
                     (map #(if % 1.0 0.0)))
          anglican (->>
                    (doquery :smc (query []
                                         (let [data [1.1 2.1 2.0 1.9 0.0 -0.1 -0.05]
                                               likes (map (fn [_] (let [mu (sample (normal 0.0 10.0))
                                                                        sigma (sample (gamma 1.0 1.0))]
                                                                    (normal mu sigma)))
                                                          (range 3))
                                               pi (sample (dirichlet [1.0 1.0 1.0]))
                                               z-prior (discrete pi)

                                               z (map (fn [y]
                                                        (let [z (sample z-prior)]
                                                          (observe (nth likes z) y)
                                                          z))
                                                      data)]
                                           (= (first z) (second z))))
                             []
                             :number-of-particles 10000)
                    (drop 20000)
                    (map :result)
                    (take 200000)
                    (map #(if % 1.0 0.0)))]

      #_(prn "exercise 3:" (mean foppl) (mean anglican))
      (is (tol? (mean foppl)
                (mean anglican)))

      (is (tol? (std foppl)
                (std anglican))))))


(deftest exercise-4-test
  (testing "Testing program from exercise 4"
    (let [foppl (->> (metropolis-within-gibbs '((let [sprinkler true
                                                    wet-grass true
                                                    is-cloudy (sample (flip 0.5))

                                                    is-raining (if (= is-cloudy true )
                                                                 (sample (flip 0.8))
                                                                 (sample (flip 0.2)))
                                                    sprinkler-dist (if (= is-cloudy true)
                                                                     (flip 0.1)
                                                                     (flip 0.5))
                                                    wet-grass-dist (if (and (= sprinkler true)
                                                                            (= is-raining true))
                                                                     (flip 0.99)
                                                                     (if (and (= sprinkler false)
                                                                              (= is-raining false))
                                                                       (flip 0.0)
                                                                       (if (or (= sprinkler true)
                                                                               (= is-raining true))
                                                                         (flip 0.9))))]
                                                (observe sprinkler-dist sprinkler)
                                                (observe wet-grass-dist wet-grass)
                                                is-raining)))
                   (drop 10000)
                   (take 10000)
                   (map #(if % 1.0 0.0)))
          anglican (->>
                    (doquery :smc (query []
                                         (let [sprinkler true
                                               wet-grass true
                                               is-cloudy (sample (flip 0.5))

                                               is-raining (if (= is-cloudy true )
                                                            (sample (flip 0.8))
                                                            (sample (flip 0.2)))
                                               sprinkler-dist (if (= is-cloudy true)
                                                                (flip 0.1)
                                                                (flip 0.5))
                                               wet-grass-dist (if (and (= sprinkler true)
                                                                       (= is-raining true))
                                                                (flip 0.99)
                                                                (if (and (= sprinkler false)
                                                                         (= is-raining false))
                                                                  (flip 0.0)
                                                                  (if (or (= sprinkler true)
                                                                          (= is-raining true))
                                                                    (flip 0.9))))]
                                           (observe sprinkler-dist sprinkler)
                                           (observe wet-grass-dist wet-grass)
                                           is-raining))
                             []
                             :number-of-particles 10000) 
                    (drop 10000)
                    (map :result)
                    (take 10000)
                    (map #(if % 1.0 0.0)))]

      #_(prn "exercise 4:" (mean foppl) (mean anglican))
      (is (tol? (mean foppl)
                (mean anglican)))

      (is (tol? (std foppl)
                (std anglican))))))

(defdist dirac [x]
  (sample* [this] x)
  (observe* [this value]
            (if (= value x)
              0
              (- (/ 1.0 0.0)))))


#_(deftest exercise-5-test
  (testing "Testing program from exercise 5"
    (let [foppl (->> (metropolis-within-gibbs '((let [x (sample (normal 0 10))
                                                    y (sample (normal 0 10))]
                                                (observe (dirac (+ x y)) 7)
                                                [x y])))
                   (drop 10000)
                   (take 10000))
          anglican (->>
                    (doquery :smc (with-primitive-procedures [dirac]
                                    (query []
                                           (let [x (sample (normal 0 10))
                                                 y (sample (normal 0 10))]
                                             (observe (dirac (+ x y)) 7)
                                             [x y])))
                             []
                             :number-of-particles 10000) 
                    (drop 10000)
                    (map :result)
                    (take 10000))]

      (is (tol? (mean (map first foppl)) (mean (map first anglican))))
      (is (tol? (mean (map second foppl)) (mean (map second anglican))))

      (is (tol? (std (map first foppl)) (std (map first anglican))))
      (is (tol? (std (map second foppl)) (std (map second anglican)))))))


