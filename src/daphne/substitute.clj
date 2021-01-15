(ns daphne.substitute)


(defn dispatch-substitute
  [exp _ _]
  (cond (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'foreach))
        :foreach

        (list? exp)
        :list

        (seq? exp)
        :seq

        (vector? exp)
        :vector

        (symbol? exp)
        :symbol

        (map? exp)
        :map

        :else :unrelated))

(defmulti substitute dispatch-substitute)

(defmethod substitute :symbol [exp sym val]
  (if (= exp sym) val exp))

(defmethod substitute :map [exp sym val]
  (into {}
        (map
         (fn [[k v]]
           [(substitute k sym val)
            (substitute v sym val)])
         (seq exp))))

(defmethod substitute :list [exp sym val]
  (apply list (map #(substitute % sym val) exp)))

(defmethod substitute :seq [exp sym val]
  (map #(substitute % sym val) exp))

(defmethod substitute :vector [exp sym val]
  (mapv #(substitute % sym val) exp))

(defmethod substitute :unrelated [exp _ _]
  exp)

(defmethod substitute :let [exp sym val]
  (let [[_ [s v] e] exp]
    (if (= s sym)
      exp ;; symbol shadowed now
      (list 'let [s (substitute v sym val)]
            (substitute e sym val)))))

(defmethod substitute :foreach
  [exp sym val]
  (let [[_ counter bindings & rest] exp]
    (apply list 'foreach
           (substitute counter sym val)
           (vec (mapcat (fn [[k v]]
                          [k (substitute v sym val)])
                        (partition 2 bindings)))
           (map #(substitute % sym val) rest))))
