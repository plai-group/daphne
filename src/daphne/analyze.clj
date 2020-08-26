(ns daphne.analyze
  (:require [clojure.set :as set]
            [daphne.substitute :refer [substitute]]
            [daphne.free-vars :refer [free-vars]]
            [daphne.desugar-let :refer [desugar-let]]
            [daphne.primitives]
            [daphne.gensym :refer [*my-gensym*]]
            [daphne.partial-evaluation :refer [fixed-point-simplify]]))


(def ^:dynamic *primitive-procedures*
  "primitive procedures for Anglican semantics" ;; TODO check implications of this choice
  (let [;; higher-order procedures cannot be primitive
        exclude '#{loop
                   map reduce
                   filter keep keep-indexed remove
                   repeatedly
                   every? not-any? some
                   every-pred some-fn
                   comp juxt partial}
        ;; runtime namespaces
        runtime-namespaces '[clojure.core anglican.runtime daphne.primitives]]
    (set (keep (fn [[k v]]
                 (when (and (not (exclude k))
                            (fn? (var-get v)))
                   k))
               (mapcat ns-publics runtime-namespaces)))))

(def ^:dynamic *bound*
  (into *primitive-procedures* ['if 'let]))

(def empty-env {})

(def empty-graph {:V #{}
                  :A {}
                  :P {}
                  :Y {}})

(defn merge-graphs
  "Defined on Page 62."
  ([a] a)
  ([a b]
   (let [{V1 :V A1 :A P1 :P Y1 :Y} a
         {V2 :V A2 :A P2 :P Y2 :Y} b]
     {:V (set/union V1 V2)
      :A (merge-with set/union A1 A2)
      :P (merge P1 P2)
      :Y (merge Y1 Y2)
      }))
  ([a b & more]
   (reduce merge-graphs
           a
           (concat [b] more))))


(defn dispatch-analyze
  "rho is environment, phi whether we are on the control flow path, exp current
  expression"
  [rho phi exp]
  (cond (or (number? exp)
            (nil? exp)
            (string? exp)
            (boolean? exp)
            (keyword? exp))
        :constant

        (symbol? exp)
        :variable

        (and (list? exp)
             (= (first exp) 'defn))
        :defn

        (and (list? exp)
             (= (first exp) 'let))
        :let

        (and (list? exp)
             (= (first exp) 'if))
        :if

        (and (list? exp)
             (= (first exp) 'sample))
        :sample

        (and (list? exp)
             (= (first exp) 'observe))
        :observe

        (list? exp)
        :application

        (seq? exp)
        :seq

        (map? exp)
        :map

        (vector? exp)
        :vector

        :else
        (throw (ex-info "Not supported." {:exp exp
                                          :rho rho
                                          :phi phi}))))

(defn third [x] (nth x 2))


(defmulti analyze dispatch-analyze)


(defmethod analyze :constant
  [rho phi c]
  [rho empty-graph c])


(defmethod analyze :variable
  [rho phi v]
  [rho empty-graph v])

(defmethod analyze :vector
  [rho phi v]
  (let [res (map #(analyze rho phi %) v)]
    [(apply merge (map first res))
     (apply merge-graphs empty-graph (map second res))
     (mapv third res)]))

(defmethod analyze :seq
  [rho phi v]
  (let [res (map #(analyze rho phi %) v)]
    [(apply merge (map first res))
     (apply merge-graphs empty-graph (map second res))
     (map third res)]))

(defmethod analyze :map
  [rho phi v]
  (let [kres (map #(analyze rho phi %) (keys v))
        vres (map #(analyze rho phi %) (vals v))]
    [(apply merge (concat (map first kres) (map first vres)))
     (apply merge-graphs empty-graph (concat (map second kres) (map second vres)))
     (into {} (->> (interleave (map third kres)
                             (map third vres))
                 (partition 2)
                 (map vec)))]))


(defmethod analyze :let
  [rho phi exp]
  (let [[_ [v e1] & body] (desugar-let exp)
        [rho1 G1 E1] (analyze rho phi e1)
        res (map #(analyze rho phi (fixed-point-simplify (substitute % v E1))) body)
        rhos (map first res)
        Gs (map second res)
        Es (map third res)]
    [(apply merge rhos)
     (apply merge-graphs G1 Gs)
     (last Es)]))


(defmethod analyze :if
  [rho phi exp]
  (let [[_ condition then else] exp
        [rho1 G1 E1] (analyze rho phi condition)
        [rho2 G2 E2] (analyze rho (fixed-point-simplify (list 'and phi E1)) then)
        [rho3 G3 E3] (analyze rho (fixed-point-simplify (list 'and phi (list 'not E1))) else)]
    [(merge rho1 rho2 rho3) (merge-graphs G1 G2 G3) (fixed-point-simplify (list 'if E1 E2 E3))]))


(defmethod analyze :sample
  [rho phi exp]
  (let [[_ e] exp
        e (fixed-point-simplify e)
        [rho {:keys [V A P Y]} E] (analyze rho phi e)
        v (*my-gensym* "sample")
        Z (free-vars E *bound*)
        F (list 'sample* (fixed-point-simplify E))]
    [rho
     {:V (conj V v)
      :A (into A (for [z Z] [z #{v}]))
      :P (assoc P v F)
      :Y Y}
     v]))

(comment
  (analyze empty-env false '(let [x (sample (normal 0 1))]
                              (let [y (sample (normal 1 2))]
                                (sample (normal x y))))))

(defmethod analyze :observe
  [rho phi exp]
  (let [[_ e obs] exp
        e (fixed-point-simplify e)
        obs (fixed-point-simplify obs)
        [rho1 G1 E1] (analyze rho phi e)
        [rho2 G2 E2] (analyze rho phi obs)
        {:keys [V A P Y]} (merge-graphs G1 G2)
        v (*my-gensym* "observe")
        F1 (list 'observe* e obs)
        F (fixed-point-simplify (list 'if phi F1 1))
        Z (free-vars e *bound*) #_(disj (free-vars e *bound*) v)
        _ (when (not (empty? (free-vars obs *bound*)))
            (throw (ex-info "Observation references random vars."
                            {:obs obs
                             :exp exp})))
        B (for [z Z] [z #{v}])]
    [(merge rho1 rho2)
     {:V (conj V v)
      :A (into A B)
      :P (assoc P v F)
      :Y (assoc Y v E2)}
     v]))


(defmethod analyze :application
  [rho phi exp]
  (let [[op & args] (if (= 'apply (first exp))
                      (apply list (second exp) (third exp))
                      exp)
        nargs (map #(analyze rho phi %) args)
        rhos (map first nargs)
        GS (map second nargs)
        ES (map third nargs)]
    (cond (rho op)
          ;; TODO check graph
          (let [[_ bindings body] (rho op)
                _ (when-not (= (count args)
                               (count bindings))
                    (throw (ex-info "Trying to apply function with wrong number of args."
                                    {:type :arity-mismatch
                                     :exp exp})))
                [rho G E] (analyze rho phi
                                   (fixed-point-simplify
                                    (list 'let (vec (interleave bindings ES)) body)))]
            [(apply merge rho rhos)
             (apply merge-graphs G GS)
             E])

          :else
          [(apply merge rhos)
           (apply merge-graphs GS)
           (fixed-point-simplify (apply list (concat (list op) ES)))]
          )))


;; TODO support multiple body expressions
(defmethod analyze :defn
  [rho phi exp]
  (let [[_ name bindings body] exp]
    [(assoc rho name (list 'fn bindings body)) empty-graph name]))


