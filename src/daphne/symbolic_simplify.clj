(ns daphne.symbolic-simplify
  (:require [anglican.runtime]
            [daphne.primitives :refer [append]]
            [clojure.core.memoize :as memoize]))

(def whitelist {'rest rest
                'first first
                'second second
                'last last
                'nth nth
                'conj conj
                'cons cons
                'vector vector
                'subvec subvec
                'get get
                'put assoc
                'append append
                'count count
                'concat concat
                'flatten (fn [x] (vec (flatten x)))
                })


(defn anglican-distribution? [sym]
  (when (symbol? sym)
    (when-let [v ((ns-publics 'anglican.runtime)
                  (symbol (str "map->"
                               (name sym)
                               "-distribution")))]
      (extends? anglican.runtime/distribution
                (type (v {}))))))

(comment
  (anglican-distribution? 'normals))


(defn dispatch-simplify [exp]
  (cond (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (and (list? exp)
             (= (first exp) 'defn))
        :defn

        (and (list? exp)
             (= (first exp) 'loop))
        :loop

        (and (list? exp)
             (= (first exp) 'foreach))
        :foreach

        (and (list? exp)
             (whitelist
              (first exp)))
        :application

        (and (list? exp)
             (anglican-distribution? (first exp)))
        :anglican-application

        (and (list? exp)
             (#{'sample 'observe} (first exp)))
        :anglican-application

        (list? exp)
        :list

        (vector? exp)
        :vector

        (map? exp)
        :map

        (seq? exp)
        :seq

        :else
        :unrelated))


(defmulti symbolic-simplify dispatch-simplify)

(def mem-symbolic-simplify (memoize/lu symbolic-simplify :lu/threshold 10000))

(defmethod symbolic-simplify :let
  [exp]
  (let [[_ bindings & body] exp]
    (apply list
           (concat (list 'let (vec
                               (mapcat (fn [[s v]] [s (mem-symbolic-simplify v)])
                                       (partition 2 bindings))))
                   (map mem-symbolic-simplify body)))))


(defmethod symbolic-simplify :if
  [exp]
  (let [[_ condition then else] exp
        new-cond (mem-symbolic-simplify condition)]
    (cond (true? new-cond)
          (mem-symbolic-simplify then)

          (false? new-cond)
          (mem-symbolic-simplify else)

          :else
          (list 'if new-cond
                (mem-symbolic-simplify then)
                (mem-symbolic-simplify else)))))

(defmethod symbolic-simplify :defn
  [exp]
  (let [[_ name bindings & body] exp]
    (apply list (concat (list 'defn name bindings)
                        (map mem-symbolic-simplify body)))))

(defmethod symbolic-simplify :loop
  [exp]
  (let [[_ counter & rest] exp]
    (apply list 'loop (mem-symbolic-simplify counter) rest)))

(defmethod symbolic-simplify :foreach
  [exp]
  (let [[_ counter & rest] exp]
    (apply list 'foreach (mem-symbolic-simplify counter) rest)))


(defmethod symbolic-simplify :application
  [exp]
  (try
    (let [f (mem-symbolic-simplify (first exp))
          s (mem-symbolic-simplify (second exp))]
      (cond (= f 'vector)
            (apply vector (map mem-symbolic-simplify (rest exp)))

            (list? s)
            (apply list
                   (conj (map mem-symbolic-simplify (rest exp))
                         f))

            (or (vector? s)
                (map? s)
                (seq? s))
            (if-let [fs (whitelist f)]
              ;; HACK guard against lookups of symbols etc, needs proper interpretation
              (if-let [res
                       (apply fs s (map mem-symbolic-simplify (rest (rest exp))))]
                res
                (apply list
                       (conj (map mem-symbolic-simplify (rest exp))
                             f)))
              (apply list
                     (conj (map mem-symbolic-simplify (rest exp))
                           f)))

            #_(seq? s)
            #_(do
              #_(println "simplifying seq" f s #_(second exp))
              (apply (whitelist f) s (map mem-symbolic-simplify (rest (rest exp)))))

            :else
            (apply list
                   (conj (map mem-symbolic-simplify (rest exp))
                         f))))
    (catch Exception _
      (try
        (apply list
               (conj (map mem-symbolic-simplify (rest exp))
                     (first exp)))
        (catch Exception _
          exp)))))


(defmethod symbolic-simplify :anglican-application
  [exp]
  (apply list (conj (map mem-symbolic-simplify (rest exp))
                    (first exp))))

(defmethod symbolic-simplify :vector
  [exp]
  (mapv mem-symbolic-simplify exp))

(defmethod symbolic-simplify :list
  [exp]
  (apply list
         (map mem-symbolic-simplify exp)))

(defmethod symbolic-simplify :seq
  [exp]
  (map mem-symbolic-simplify exp))

(defmethod symbolic-simplify :map
  [exp]
  (into {}
        (map (fn [[k v]]
               [(mem-symbolic-simplify k)
                (mem-symbolic-simplify v)])
             exp)))

(defmethod symbolic-simplify :unrelated
  [exp]
  exp)


