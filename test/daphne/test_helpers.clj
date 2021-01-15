(ns daphne.test-helpers)

(defn local-gensym []
  (let [gcounter (atom 0)]
    (fn [s]
      (symbol (str s (swap! gcounter inc))))))

(defn err? [a b] (> (Math/abs (- a b)) 5e-2))


