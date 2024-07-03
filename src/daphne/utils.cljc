(ns daphne.utils)

(defn value?
  "Naive value check, rules out expressions containing symbols."
  [x]
  (if (or (seq? x)
          (vector? x)
          (map? x)
          (set? x))
    (not (some symbol? (flatten x)))
    (not (symbol? x))))

