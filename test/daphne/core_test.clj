(ns daphne.core-test
  (:require [clojure.test :refer [deftest testing is]]
            [daphne.core :refer :all]))


(comment

  (count-graph
   '((defn observe-data [_ data slope bias]
       (rest (rest data)))
     (let [data (vector 1.0 2.1 2.0 3.9 3.0 5.3
                        4.0 7.7 5.0 10.2 6.0 12.9)]
       (loop 6 data observe-data slope bias)
       ))) 

  (def prog 
    (code->graph '((defn hmm-step [t states data trans-dists likes]
                    (let [z (sample (get trans-dists
                                         (last states)))]
                      (observe (get likes z)
                               (get data t))
                      (append states z)))
                  (let [data [0.9 0.8 0.7 0.0 -0.025 -5.0 -2.0 -0.1
                              0.0 0.13 0.45 6 0.2 0.3 -1 -1]
                        trans-dists [(discrete [0.10 0.50 0.40])
                                     (discrete [0.20 0.20 0.60])
                                     (discrete [0.15 0.15 0.70])]
                        likes [(normal -1.0 1.0)
                               (normal 1.0 1.0)
                               (normal 0.0 1.0)]
                        states [(sample (discrete [0.33 0.33 0.34]))]]
                    (loop 16 states hmm-step data trans-dists likes))))) 


  (clojure.pprint/pprint
   (:A (second prog))) 

  (count-graph
   '((defn hmm-step [t states data trans-dists likes]
       (let [z (sample (get trans-dists
                            (last states)))]
         (observe (get likes z)
                  (get data t))
         (append states z)))
     (let [data [0.9 0.8 0.7 0.0 -0.025 -5.0 -2.0 -0.1
                 0.0 0.13 0.45 6 0.2 0.3 -1 -1]
           trans-dists [(discrete [0.10 0.50 0.40])
                        (discrete [0.20 0.20 0.60])
                        (discrete [0.15 0.15 0.70])]
           likes [(normal -1.0 1.0)
                  (normal 1.0 1.0)
                  (normal 0.0 1.0)]
           states [(sample (discrete [0.33 0.33 0.34]))]]
       (loop 16 states hmm-step data trans-dists likes)))) 

  (count-graph
   '((defn observe-data [_ data slope bias]
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

  (let [slope (sample (normal 0.0 10.0))]
    (let [bias (sample (normal 0.0 10.0))]
      (let [data [1.0 2.1 2.0 3.9 3.0 5.3 4.0 7.7 5.0 10.2 6.0 12.9]]
        (let [dontcare2330543
              (let [a2330544 slope]
                (let [a2330545 bias]
                  (let [acc2330546
                        (let [_ 0]
                          (let [data data]
                            (let [slope a2330544]
                              (let [bias a2330545]
                                (let [xn 1.0]
                                  (let [yn 2.1]
                                    (let [zn (+ (* slope xn) bias)]
                                      (let [dontcare2330574 (observe (normal zn 1.0) yn)]
                                        (2.0 3.9 3.0 5.3 4.0 7.7 5.0 10.2 6.0 12.9)))))))))]
                    (let [acc2330547
                          (let [_ 1]
                            (let [data acc2330546]
                              (let [slope a2330544]
                                (let [bias a2330545]
                                  (let [xn (first data)]
                                    (let [yn (second data)]
                                      (let [zn (+ (* slope xn) bias)]
                                        (let [dontcare2330575 (observe (normal zn 1.0) yn)]
                                          (rest (rest data))))))))))]
                      acc2330547))))]
          [slope bias]))))
  )


(deftest graph-tests
  (testing "Testing compiled graph metrics."

    (is (= [3 2]
           (count-graph
            '((let [mu (sample (normal 1 (sqrt 5)))
                    sigma (sqrt 2)
                    lik (normal mu sigma)]
                (observe lik 8)
                (observe lik 9)
                mu)))))

    (is (= [8 12]
           (count-graph
            '((defn observe-data [_ data slope bias]
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
                (vector slope bias))))))

    (is (= [33 32]
           (count-graph
            '((defn hmm-step [t states data trans-dists likes]
                (let [z (sample (get trans-dists
                                     (last states)))]
                  (observe (get likes z)
                           (get data t))
                  (append states z)))
              (let [data [0.9 0.8 0.7 0.0 -0.025 -5.0 -2.0 -0.1
                          0.0 0.13 0.45 6 0.2 0.3 -1 -1]
                    trans-dists [(discrete [0.10 0.50 0.40])
                                 (discrete [0.20 0.20 0.60])
                                 (discrete [0.15 0.15 0.70])]
                    likes [(normal -1.0 1.0)
                           (normal 1.0 1.0)
                           (normal 0.0 1.0)]
                    states [(sample (discrete [0.33 0.33 0.34]))]]
                (loop 16 states hmm-step data trans-dists likes))))))

    (is (= [146 705]
           (count-graph '((let [weight-prior (normal 0 1)
                                W_0 (foreach 10 []
                                             (foreach 1 [] (sample weight-prior)))
                                W_1 (foreach 10 []
                                             (foreach 10 [] (sample weight-prior)))
                                W_2 (foreach 1 []
                                             (foreach 10 [] (sample weight-prior)))

                                b_0 (foreach 10 []
                                             (foreach 1 [] (sample weight-prior)))
                                b_1 (foreach 10 []
                                             (foreach 1 [] (sample weight-prior)))
                                b_2 (foreach 1 []
                                             (foreach 1 [] (sample weight-prior)))

                                x   (mat-transpose [[1] [2] [3] [4] [5]])
                                y   [[1] [4] [9] [16] [25]]
                                h_0 (mat-tanh (mat-add (mat-mul W_0 x)
                                                       (mat-repmat b_0 1 5)))
                                h_1 (mat-tanh (mat-add (mat-mul W_1 h_0)
                                                       (mat-repmat b_1 1 5)))
                                mu  (mat-transpose
                                    (mat-tanh (mat-add (mat-mul W_2 h_1)
                                                       (mat-repmat b_2 1 5))))]
                            (foreach 5 [y_r y
                                        mu_r mu]
                                     (foreach 1 [y_rc y_r
                                                 mu_rc mu_r]
                                              (observe (normal mu_rc 1) y_rc)))
                            [W_0 b_0 W_1 b_1]))))))) 


 



