(ns daphne.invert-test
  (:require [daphne.invert :refer [frontier moralize min-fill
                                           new-edges faithful-inversion]]
            [loom.graph :as g]
            [clojure.test :refer [deftest testing is]]))

(deftest helpers-test
  (testing "Testing helper functions."
    (is (= (frontier (g/digraph {:A #{:D} :B #{:D} :C #{:D}}) #{:A :B :C}
                     g/predecessors)
           #{:A :B :C}))
    (is (= (moralize (g/digraph {:A #{:D} :B #{:D} :C #{:D}}))
           #loom.graph.BasicEditableGraph{:nodeset #{:A :D :B :C},
                                          :adj {:A #{:D :B :C},
                                                :D #{:A :B :C},
                                                :B #{:A :D :C},
                                                :C #{:A :D :B}},
                                          :attrs nil}))
    (is (= (new-edges (g/graph {:A #{:B :D} :B #{:C} :C #{:D}})
                      :A
                      #{})
           '(#{:D :B})))

    (is (= (min-fill #{:A :B}
                     (g/graph {:A #{:B :D} :B #{:C} :C #{:D}})
                     #{})
           [:B '(#{:A :C})]))))


(deftest faithful-inversion-test
  (testing "Faithful inversion algorithm (NaMI)"
    (is (= (faithful-inversion (g/digraph {:A #{:B :C} :B #{:D} :C #{:E}})
                               #{:A :B :C}
                               true)
           #loom.graph.BasicEditableDigraph{:nodeset #{:A :D :B :C :E},
                                            :adj {:B #{:A :C},
                                                  :C #{:A},
                                                  :E #{:B :C},
                                                  :D #{:B}},
                                            :in {:A #{:B :C},
                                                 :C #{:B :E},
                                                 :B #{:D :E}}}))
    (is (= (faithful-inversion (g/digraph {:A #{:C} :B #{:C} :C #{:D}})
                               #{:A :B :C}
                               true)
           #loom.graph.BasicEditableDigraph{:nodeset #{:A :D :B :C},
                                            :adj {:A #{:B}, :C #{:A :B}, :D #{:C}},
                                            :in {:B #{:A :C}, :A #{:C}, :C #{:D}}}))
    (is (= (faithful-inversion (g/digraph {:D #{:G}
                                           :I #{:G :S}
                                           :G #{:H :L}
                                           :S #{:J}
                                           :L #{:J}
                                           :J #{:H}})
                               #{:D :I :G :S :L}
                               true)
           #loom.graph.BasicEditableDigraph{:nodeset #{:L :I :D :J :G :H :S},
                                            :adj {:I #{:D},
                                                  :G #{:I :D :S},
                                                  :S #{:I},
                                                  :L #{:G :S},
                                                  :J #{:L :G :S},
                                                  :H #{:L :G}},
                                            :in {:D #{:I :G},
                                                 :I #{:G :S},
                                                 :S #{:L :J :G},
                                                 :G #{:L :J :H},
                                                 :L #{:J :H}}}))))



(comment
  (def tree-graph (g/digraph {:x0 #{:x1 :x2} :x1 #{:x3 :x4} :x2 #{:x5 :x6}})) 



  (def tree-graph (g/digraph {:z0 #{:z2 :z3}
                              :z1 #{:z3 :z2}
                              :z3 #{:x0 :z5}
                              :z4 #{:z5}
                              :z5 #{:x1}})) 

  (require '[loom.io :as lio]) 

  (lio/view
   (faithful-inversion tree-graph
                       #{:z0 :z1 :z2 :z3 :z4 :z5}
                       true)) 


  (lio/view
   (faithful-inversion (g/digraph {:D #{:G}
                                   :I #{:G :S}
                                   :G #{:H :L}
                                   :S #{:J}
                                   :L #{:J}
                                   :J #{:H}})
                       #{:D :I :G :S :L}
                       false)))
