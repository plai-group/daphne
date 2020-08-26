(ns daphne.invert
  "Faithful graphical model inversion due to https://arxiv.org/abs/1712.00287."
  (:require [loom.graph :as g]
            [clojure.set :as set]))


(defn- all-pairs [nodes]
  (for [n nodes
        o nodes
        :when (not= o n)]
    [n o]))

(defn moralize [g]
  (let [ug (g/graph g)]
    (reduce (fn [g parents]
              (reduce g/add-edges
                      g
                      (all-pairs parents)))
            ug
            (map (partial g/predecessors g) (g/nodes g)))))


(defn frontier [g latents upstream]
  (set
   (filter (fn [n] (zero? (count (filter latents (upstream g n)))))
           latents)))

(defn new-edges [I v marked]
  (let [neighbors (filter (comp not marked)
                          (g/successors I v))]
    (distinct
     (for [v neighbors
           u neighbors
           :when (and (not= u v)
                      (not ((g/successors I v) u)))]
       #{v u}))))


(defn min-fill [frontier g marked]
  (let [candidates (map (fn [v]
                          [v (new-edges g v marked)])
                        frontier)]
    (apply min-key (comp count second) candidates)))


(defn all-upstream-marked? [upstream g marked u latents]
  (set/subset? (set (filter latents (upstream g u))) marked))


(defn faithful-inversion
  "NaMI Graph Inversion algorithm.

  Algorithm 1 of https://arxiv.org/abs/1712.00287."
  [G Z top-mode?]
  (let [I (moralize G)
        marked #{}
        H (apply g/digraph (g/nodes G))
        upstream (if top-mode? g/predecessors g/successors)
        downstream (if top-mode? g/successors g/predecessors)
        S (frontier G Z upstream)]
    (loop [S S
           H H
           I I
           marked marked]
      #_(println S H I marked)
      (if-not (empty? S)
        (let [[v edges] (min-fill S I marked)
              I (apply g/add-edges I (map vec edges))
              new-neighbors (filter (comp not marked)
                                    (g/successors I v))
              H (apply g/add-edges H (for [u new-neighbors]
                                       [u v]))
              marked (conj marked v)
              S (disj S v)
              S (->> (downstream G v)
                     (filter (partial all-upstream-marked? upstream G marked Z))
                     (filter Z)
                     (into S))]
          (recur S H I marked))
        H))))


(defn faithful-program-inversion
  "Takes a program and adds the minimal, faithfully inverted graph as H."
  [{:keys [V A Y] :as prog}]
  (assoc prog :H
         (faithful-inversion (g/digraph A)
                             (set (filter (comp not (set (keys Y))) V))
                             true)))



