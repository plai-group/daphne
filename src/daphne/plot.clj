(ns daphne.plot
  "Plotting routines for the graph."
  (:require [loom.graph :as g]
            [loom.io :as lio]
            [clojure.java.io :as io]
            [daphne.core :as fc]
            [daphne.invert :refer [faithful-program-inversion]]))


(def ^:dynamic *gensyms*)

(def ^:dynamic *offset* 0)

(defn- plot-gensym
  "Generates symbol and integer gensyms to refer to tensor positions later."
  [sym]
  ;; filter out all internal gensyms
  (if-not (#{"sample" "observe"} sym)
    (gensym sym)
    (let [f (first @*gensyms*)]
      (swap! *gensyms* rest)
      (if (= sym "observe")
        (symbol (str "y"  (- f *offset*)))
        (symbol (str "x" f))))))

(defn- byte-spit [fn bs]
  (with-open [out (io/output-stream (io/file fn))]
    (.write out bs)))


(defn plot-graph [code path+file-prefix]
  (binding [*gensyms* (atom (range))
            daphne.gensym/*my-gensym* plot-gensym]
    (let [[rho G E] (fc/code->graph code)
          H (:H (faithful-program-inversion G))
          A (g/digraph (:A G))]
      (if-not path+file-prefix
        (do
          (lio/view H)
          (lio/view A))
        (do
          (byte-spit (str path+file-prefix "_inv.pdf")
                     (lio/render-to-bytes H :fmt :pdf))
          (byte-spit (str path+file-prefix "_gen.pdf")
                     (lio/render-to-bytes A :fmt :pdf)))))))


(comment
  (def normal-normal-prog
    '((let [mu (sample (normal 1 (sqrt 5)))
            sigma (sqrt 2)
            lik (normal mu sigma)]
        (observe lik 8)
        (observe lik 9)
        mu))) 


  (plot-graph normal-normal-prog "/tmp/normal_normal") 
  )
