(ns daphne.command
  (:refer-clojure :exclude [compile])
  (:require [clojure.string :as str]
            [clojure.edn :as edn]
            [clojure.tools.cli :refer [parse-opts]]
            [clojure.java.io :as io]
            [clojure.data.json :as json]
            [daphne.gensym :refer [*my-gensym*]]
            [daphne.hy :refer [foppl->python]]
            [daphne.desugar :refer [desugar]]
            [daphne.desugar-datastructures :refer [desugar-datastructures]]
            [daphne.metropolis-within-gibbs :refer [metropolis-within-gibbs]]
            [daphne.hmc :refer [hmc]]
            [daphne.core :refer [program->graph]])
  (:import [java.io PushbackReader StringReader])
  (:gen-class))

;; This file is following https://github.com/clojure/tools.cli

(defn usage [options-summary]
  (->> ["This is the daphne probabilistic program compiler."
        ""
        "Usage: daphne [options] action"
        ""
        "Options:"
        options-summary
        ""
        "Actions:"
        "  graph        Create graphical model of the program"
        "  desugar      Return a desugared syntax object of the program"
        "  python-class Create a Python class with sample and log probability methods for the program"
        "  infer        Run inference on the program"
        ""
        "Please refer to the manual page for more information."]
       (str/join \newline)))

(defn error-msg [errors]
  (str "The following errors occurred while parsing your command:\n\n"
       (str/join \newline errors)))

(def actions #{"graph" "desugar" "python-class" "infer"})

(def cli-options
  ;; An option with a required argument
  (let [formats #{:json :edn}
        algorithms #{:hmc :metropolis-within-gibbs}]
    [["-f" "--format FORMAT" "Output format"
      :default :json
      :parse-fn keyword
      :validate [formats (str "Must be one of: " (str/join ", " formats))]]
     ["-s" "--source SOURCECODE" "Program source code, by default STDIN is used."
      :default nil
      :validate [read-string "Cannot read program code."]]
     ["-i" "--input-file SOURCEFILE" "Program source file, by default STDIN is used."
      :default nil
      :validate [#(.exists (io/file %))
                 "Program source file does not exist."]]
     [nil "--num-samples N" "Number of samples to draw"
      :default 1000
      :parse-fn #(Integer/parseInt %)
      :validate [pos? "Number of samples must be positive."]]
     ["-a" "--algorithm ALGORITHM" "Inference algorithm to use"
      :default :metropolis-within-gibbs
      :parse-fn keyword
      :validate [algorithms (str "Algorithm must be one of: "
                                 (str/join ", " algorithms))]]
     ["-o" "--output-file OUTPUTFILE"
      "File to write the output to, otherwise STDOUT is used."
      :default nil
      :validate [#(io/file %) "Output file is not a valid file name."]]
     ;; A non-idempotent option (:default is applied first)
     ["-v" nil "Verbosity level"
      :id :verbosity
      :default 0
      :update-fn inc]
     ;; A boolean option defaulting to nil
     ["-h" "--help"]]))

(defn validate-args
  "Validate command line arguments. Either return a map indicating the program
  should exit (with a error message, and optional ok status), or a map
  indicating the action the program should take and the options provided."
  [args]
  (let [{:keys [options arguments errors summary]} (parse-opts args cli-options)]
    (cond
      (:help options) ; help => exit OK with usage summary
      {:exit-message (usage summary) :ok? true}

      errors          ; errors => exit with description of errors
      {:exit-message (error-msg errors)}

      ;; custom validation on arguments
      (and (:source options) (:source-file options))
      {:exit-message "The --source and --source-file options are exclusive."}

      (and (= 1 (count arguments))
           (actions (first arguments)))
      {:action (keyword (first arguments)) :options options}

      (not (actions (first arguments)))
      {:exit-message (str "Unknown command, must be one of: "
                          (str/join ", " actions))}

      :else           ; failed custom validation => exit with usage summary
      {:exit-message (usage summary)})))

(defn exit [status msg]
  (println msg)
  (System/exit status))

(defn read-all-exps [s]
  (try
    (with-open [in (PushbackReader. (StringReader. s))]
      (let [edn-seq (repeatedly (partial edn/read {:eof :theend} in))]
        (apply list (take-while (partial not= :theend) edn-seq))))
    (catch Exception e
      (throw (ex-info "Cannot read symbolic expressions from string."
                      {:string s
                       :exception e})))))

(defn infer [code opts]
  (let [algo ({:metropolis-within-gibbs metropolis-within-gibbs
               :hmc                     hmc} (:algorithm opts))]
    (->> code
       algo
       (take (:num-samples opts))
       vec)))

(defn calculate [action code opts]
  (when (pos? (:verbosity opts))
    (println "Executing" action " for:")
    (apply println code))
  (let [gensyms (atom (range))]
    (binding [*my-gensym* (fn [s]
                            (let [f (first @gensyms)]
                              (swap! gensyms rest)
                              (symbol (str s f))))]
      (case action
        :graph (program->graph code)
        :desugar (-> code desugar desugar-datastructures)
        :python-class (foppl->python code)
        :infer (infer code opts)))))

(defn -main [& args]
  (let [{:keys [action options exit-message ok?]} (validate-args args)]
    (if exit-message
      (exit (if ok? 0 1) exit-message)
      (let [source (or (:source options)
                       (slurp (or (:input-file options) *in*)))
            code (read-all-exps source)
            out' (calculate action code options)
            out (if (not (string? out'))
                  (case (:format options)
                    :json (json/json-str out')
                    :edn  (pr out'))
                  out')]
        (when (pos? (:verbosity options))
          (println))
        (if (:output-file options)
          (spit (:output-file options) out)
          (println out))
        (System/exit 0)))))

