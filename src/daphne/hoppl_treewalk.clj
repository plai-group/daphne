(ns daphne.hoppl-treewalk)

(def empty-env {::parent nil
                '+       +})

(defn lookup [env s syntax position]
  (let [v (get env s ::not-found)]
    (cond (not= v ::not-found)
          v

          (not (nil? (::parent env)))
          (lookup (::parent env) s syntax position)

          :else
          (throw (ex-info "Name not found in scope." {:name     s
                                                      :syntax   syntax
                                                      :position position})))))

(defn value? [exp]
  (or (number? exp)
      (string? exp)
      (keyword? exp)
      (boolean? exp)))

(defn primitive? [f]
  (not (and (vector? f) (= (first f) 'fn))))

(declare eval)

(defn apply [env syntax position]
  (let [callsite (get-in syntax position)
        f (eval env syntax (conj position 0))
        arg-values (map (partial eval env syntax)
                        (for [i (range 1 (count callsite))]
                          (conj position i)))]
    (if (primitive? f)
      (clojure.core/apply f arg-values)
      (let [arg-names (second f)
            new-env   (->> (interleave arg-names arg-values)
                         (partition 2)
                         (map vec)
                         (into {}))
            new-env   (assoc new-env ::parent env)]
        ;; eval body
        (eval new-env syntax (conj (conj position 0) 2))))))

(defn eval [env syntax position]
  (let [exp (get-in syntax position)]
    (cond (value? exp)
          exp

          (symbol? exp)
          (lookup env exp syntax position)

          (vector? exp)
          (cond (= (first exp) 'fn)
                exp

                (= (first exp) 'if)
                (let [p (eval env syntax (conj position 1))]
                  (if p
                    (eval env syntax (conj position 2))
                    (eval env syntax (conj position 3))))

                :else
                (apply env syntax position)))))


(comment

  (eval {'x 42} '[x] [0])

  (eval empty-env '[[+ 1 1]] [0])

  (eval empty-env '[[[fn [x] x] 0]] [0])

  (eval empty-env '[[[fn [x] [[fn [x] [+ 1 x]] 1]] 0]] [0])

  (eval empty-env '[[if true 42 43]] [0])

  (eval empty-env '[[if false 42 43]] [0])

  (eval {'x 0, :daphne.hoppl-treewalk/parent #:daphne.hoppl-treewalk{:parent nil}} [0 0 2] '[[[fn [x] x] 0]])

  )


