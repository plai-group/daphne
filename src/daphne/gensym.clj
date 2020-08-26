(ns daphne.gensym)

(def ^:dynamic *gensyms* (atom (range)))

(def ^:dynamic *my-gensym* gensym)
