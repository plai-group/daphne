(ns daphne.main
  (:require [daphne.hy :refer [foppl->python]])
  (:gen-class))

(defn -main [& args]
  (let [input-program (read-string (str "("
                                        (slurp (first args))
                                        ")"))]
    (spit (second args) (foppl->python input-program))
    (System/exit 0)))

