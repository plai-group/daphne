(ns daphne.gsdm
  "Compile Daphne probabilistic programs to GSDM dataset classes.
  
  This namespace provides compilation from Daphne's graph representation
  to Python classes compatible with the Graphical Simulator Inference (GSDM)
  framework for neural network-based inference."
  (:require [daphne.core :as core]
            [daphne.desugar :as desugar]
            [daphne.desugar-datastructures :as desugar-datastructures]
            [backtick :refer [template]]
            [clojure.pprint :refer [pprint]]
            [clojure.java.shell :as sh]))


;; =============================================================================
;; Utility Functions
;; =============================================================================
;; 
;; NOTE: These are minimal duplicates from daphne.command for standalone use.
;; When used via command.clj, the command.clj versions are used instead.

(defn- transform-vals 
  "Transform values in a map."
  [f m]
  (into {} (map (fn [[k v]] [k (f v)]) m)))

(defn desugar-datastructures-graph
  "Desugar data structures in the graph to reveal hidden samples/observes.
  
  Critical for programs using reduce/loop that hide observes in data structures."
  [[rho g return]]
  [(transform-vals (comp desugar-datastructures/desugar-datastructures 
                        desugar/desugar) rho)
   (update-in g [:P] (partial transform-vals desugar-datastructures/desugar-datastructures))
   (desugar-datastructures/desugar-datastructures return)])

;; =============================================================================
;; Graph Analysis
;; =============================================================================

(defn sample-expr?
  "Check if expression is a sample statement."
  [expr]
  (and (seq? expr)
       (= 'sample* (first expr))))

(defn observe-expr?
  "Check if expression is an observe statement."
  [expr]
  (and (seq? expr)
       (= 'observe* (first expr))))

(defn extract-dist
  "Extract distribution expression from sample* or observe* statement."
  [expr]
  (when (or (sample-expr? expr) (observe-expr? expr))
    (second expr)))

(defn extract-vertices
  "Extract variable information from graph G."
  [G]
  (for [v (:V G)]
    (let [proc ((:P G) v)
          is-observed (contains? (:Y G) v)]
      {:name v
       :proc proc
       :type (cond
               is-observed :observed
               (sample-expr? proc) :latent
               :else :deterministic)
       :dist (extract-dist proc)})))

(defn graph->var-info
  "Extract comprehensive variable and edge information from graph G."
  [G]
  (let [vertices (extract-vertices G)
        edges (for [[parent children] (:A G)
                    child children]
                [parent child])]
    {:vars vertices
     :edges edges}))

;; =============================================================================
;; Expression Translation (Daphne → Hy)
;; =============================================================================

(defn expr->hy
  "Convert Daphne expression to Hy S-expression.
  
  env is a map of variable names to their Hy representations."
  [expr env]
  (cond
    (number? expr) expr
    (true? expr) 'True
    (false? expr) 'False
    (nil? expr) 'None
    (symbol? expr) (case expr
                     true 'True
                     false 'False
                     nil 'None
                     (get env expr expr))
    (seq? expr)
    (let [op (first expr)]
      (case op
        normal `(~'Normal ~(expr->hy (second expr) env)
                          ~(expr->hy (nth expr 2) env))
        beta `(~'Beta ~(expr->hy (second expr) env)
                      ~(expr->hy (nth expr 2) env))
        bernoulli `(~'Bernoulli ~(expr->hy (second expr) env))
        gamma `(~'Gamma ~(expr->hy (second expr) env)
                        ~(expr->hy (nth expr 2) env))
        uniform `(~'Uniform ~(expr->hy (second expr) env)
                            ~(expr->hy (nth expr 2) env))
        + `(~'+ ~@(map #(expr->hy % env) (rest expr)))
        - `(~'- ~@(map #(expr->hy % env) (rest expr)))
        * `(~'* ~@(map #(expr->hy % env) (rest expr)))
        / `(~'/ ~@(map #(expr->hy % env) (rest expr)))
        exp `(~'torch.exp ~(expr->hy (second expr) env))
        ;; Default: assume function call
        `(~op ~@(map #(expr->hy % env) (rest expr)))))
    :else expr))

;; =============================================================================
;; Hy Code Generation
;; =============================================================================

(defn generate-spaces-hy
  "Generate the spaces property for GSDM dataset."
  [var-info]
  (template
    (defn [property] spaces [self]
      ~(vec (for [v (:vars var-info)]
              `(~'RealSpace ~(str (:name v))
                           (~'torch.Size [1])
                           :exchangeable "none"))))))

(defn var-name->val-name
  "Convert variable name to value variable name (sample1 -> sample1_val)."
  [var-name]
  (symbol (str (name var-name) "_val")))

(defn generate-sample-statement
  "Generate Hy statement to sample from a distribution."
  [var-info env]
  (let [var-name (:name var-info)
        val-name (var-name->val-name var-name)
        dist-expr (:dist var-info)]
    (when dist-expr
      `(~'setv ~val-name (~'.sample ~(expr->hy dist-expr env))))))

(defn generate-sample-object
  "Generate Hy code to create a Sample object."
  [var-info idx]
  (let [var-name (:name var-info)
        is-observed (= :observed (:type var-info))
        ;; For amortized inference: always sample from distribution during training
        val-expr `(~'.unsqueeze ~(var-name->val-name var-name) 0)
        state-marker (if is-observed 'OBSERVED 'LATENT)]
    `(~'Sample (~'get (~'. ~'self ~'spaces) ~idx)
               ~val-expr
               (~'* (~'torch.ones 1 :dtype ~'torch.long) ~state-marker))))

(defn generate-simulator-hy
  "Generate the run_simulator method with topologically sorted variables."
  [var-info G]
  (let [vars (:vars var-info)
        ;; Get topologically sorted variable names
        topo-order (core/topo-sort G)
        ;; Create map from var name to var info
        var-map (into {} (map (fn [v] [(:name v) v]) vars))
        ;; Sort vars by topological order
        sorted-vars (keep var-map topo-order)
        ;; For amortized inference: sample ALL variables (both latent and observed)
        all-vars-with-dist (filter #(:dist %) sorted-vars)
        ;; Build environment mapping variable names to their _val versions
        env (into {} (map (fn [v] [(:name v) (var-name->val-name (:name v))]) 
                          all-vars-with-dist))]
    (template
      (defn run_simulator [self]
        ;; Generate sample statements for ALL variables (latent + observed)
        ;; in topologically sorted order to respect dependencies
        ~@(keep #(generate-sample-statement % env) all-vars-with-dist)
        
        ;; Return list of Sample objects for ALL variables
        ~(vec (map-indexed (fn [idx v] (generate-sample-object v idx)) vars))))))

(defn generate-graph-mask-hy
  "Generate get_graphical_model_mask_and_indices method.
  
  This creates the sparse attention mask for graph attention mechanism."
  [var-info]
  (let [edges (:edges var-info)
        num-vars (count (:vars var-info))
        var-names (map :name (:vars var-info))
        name->idx (zipmap var-names (range))
        edge-stmts (for [[from to] edges]
                     (let [from-idx (name->idx from)
                           to-idx (name->idx to)]
                       `(do
                          (.add (~'get ~'indices ~from-idx) ~to-idx)
                          (.add (~'get ~'indices ~to-idx) ~from-idx))))]
    (template
      (defn get_graphical_model_mask_and_indices [self]
        ;; Initialize indices - each variable attends to itself
        (setv indices (lfor i (range ~num-vars) (set [i])))
        
        ;; Add bidirectional edges from graph
        ~@edge-stmts
        
        ;; Convert to mask and tensor format
        (setv indices-lists (lfor idx-set indices (list idx-set)))
        (setv max-attendable (max (lfor idx-list indices-lists (len idx-list))))
        
        (setv mask (torch.zeros ~num-vars max-attendable :dtype torch.long))
        (for [[i idx-list] (enumerate indices-lists)]
          (setv (get mask i (slice None (len idx-list))) 1))
        
        (setv padded-indices
              (lfor idx-list indices-lists
                    (+ idx-list (* [0] (- max-attendable (len idx-list))))))
        (setv indices-tensor (torch.tensor padded-indices :dtype torch.long))
        
        [mask indices-tensor]))))

(defn generate-gsdm-class-hy
  "Generate complete GSDM dataset class as Hy S-expression."
  [G class-name]
  (let [var-info (graph->var-info G)
        imports '((import torch)
                  (import numpy :as np)
                  (import torch.distributions [Normal Beta Bernoulli Uniform Gamma])
                  (import graphical_simulator_inference.models.measure [RealSpace Sample])
                  (import graphical_simulator_inference.models.primitives [LATENT OBSERVED])
                  (import graphical_simulator_inference.datasets.graphical_dataset [GraphicalDataset])
                  (import graphical_simulator_inference.models.measure [marginal_edges flatten_samples])
                  (import collections [OrderedDict]))]
    (template
      (do
        ~@imports
        
        (defclass ~(symbol class-name) [GraphicalDataset]
          "Auto-generated GSDM dataset from Daphne program."
          
          (defn __init__ [self config seed is_test]
            (.__init__ (super) config seed is_test)
            (setv self.states_and_edges
                  (lfor _ (range config.data.n_states)
                        (.sample_states_and_edges self 
                          :marg_prop config.data.marg_prop))))
          
          ~(generate-spaces-hy var-info)
          
          ~(generate-graph-mask-hy var-info)
          
          ~(generate-simulator-hy var-info G)
          
          (defn sample_states_and_edges [self marg_prop]
            (setv samples (.run_simulator self))
            (setv states (OrderedDict (lfor s samples 
                                            [s.space.id s.value])))
            (setv translated_edges (marginal_edges self.sparse_attention_and_indices
                                                   states))
            [states translated_edges])
          
          (defn __len__ [self]
            (if self.is_test
                self.config.data.test_dataset_length
                self.config.data.dataset_length))
          
          (defn __getitem__ [self index [marginalize True]]
            ;; Generate fresh samples for amortized inference training
            (setv samples (.run_simulator self))
            
            ;; Get cached edges (graph structure only, not values!)
            (setv [states translated_edges] 
                  (get self.states_and_edges (% index self.config.data.n_states)))
            
            ;; DO NOT overwrite sample values - use fresh samples for training
            ;; The cached states are only for graph structure consistency
            
            [(flatten_samples samples) translated_edges])
          
          (defn get_sample [self index]
            "Get a sample with proper RNG seeding for reproducibility."
            (import numpy :as np)
            (import torch)
            (np.random.seed index)
            (torch.manual_seed index)
            
            ;; Generate fresh samples with deterministic RNG
            (setv samples (.run_simulator self))
            
            ;; Get graph structure (edges only)
            (setv valid-states-and-edges 
                  (lfor _ (range self.config.data.n_states)
                        (.sample_states_and_edges self :marg_prop 0)))
            (setv [states translated-edges] 
                  (get valid-states-and-edges (% index self.config.data.n_states)))
            
            ;; DO NOT overwrite sample values - use fresh samples with seeded RNG
            
            [samples (flatten_samples samples) translated-edges])
          
          (defn avg_log_prob [self [N None]]
            "Return average log probability (not used)."
            (torch.tensor 0.0))
          
          (defn validation_metrics [self samples data model]
            {})
          
          (defn plot [self samples data prefix]
            None))))))

;; =============================================================================
;; Hy → Python Conversion
;; =============================================================================

(defn hy->python
  "Convert Hy S-expression to Python code via hy2py."
  [hy-expr]
  (let [hy-str (with-out-str (pprint hy-expr))
        result (sh/sh "hy2py" :in hy-str)]
    (if (= 0 (:exit result))
      (:out result)
      (throw (ex-info "hy2py conversion failed"
                      {:error (:err result)
                       :hy-code hy-str})))))

;; =============================================================================
;; Main Compilation Functions
;; =============================================================================

(defn gsdm
  "Compile Daphne code to GSDM Python class (for use with command.clj).
  
  Takes code as Clojure data structure (not file path).
  Returns Python source code as string.
  
  Options can include :class-name (default: 'GeneratedDataset')"
  ([code] (gsdm code {}))
  ([code opts]
   (let [class-name (get opts :class-name "GeneratedDataset")
         [_ G _] (-> code
                     core/program->graph
                     desugar-datastructures-graph)
         hy-code (generate-gsdm-class-hy G class-name)]
     (hy->python hy-code))))

(defn compile-gsdm
  "Compile Daphne program to GSDM dataset class (standalone file-based interface).
  
  Args:
    daphne-file - Path to .daphne source file
    output-file - Path to output .py file
    
  Options:
    :class-name - Name for generated class (default: GeneratedDataset)
    
  Returns:
    Map with :graph, :hy-code, :python-file"
  [daphne-file output-file & {:keys [class-name]
                               :or {class-name "GeneratedDataset"}}]
  (let [code (with-open [in (java.io.PushbackReader. 
                            (java.io.StringReader. (slurp daphne-file)))]
               (let [edn-seq (repeatedly (partial clojure.edn/read {:eof :theend} in))]
                 (apply list (take-while (partial not= :theend) edn-seq))))
        [_ G _] (-> code
                    core/program->graph
                    desugar-datastructures-graph)
        hy-code (generate-gsdm-class-hy G class-name)
        python-code (hy->python hy-code)]
    (spit output-file python-code)
    {:graph G
     :hy-code hy-code
     :python-file output-file
     :num-vars (count (:V G))
     :num-edges (count (apply concat (vals (:A G))))}))

(defn -main
  "CLI entry point for GSDM compiler."
  [& args]
  (if (< (count args) 2)
    (do
      (println "Usage: clj -M -m daphne.gsdm <input.daphne> <output.py> [class-name]")
      (println)
      (println "Compile Daphne probabilistic program to GSDM dataset class.")
      (println)
      (println "Arguments:")
      (println "  input.daphne  - Input Daphne source file")
      (println "  output.py     - Output Python file")
      (println "  class-name    - Optional class name (default: GeneratedDataset)")
      (System/exit 1))
    (let [[input output class-name] args
          result (compile-gsdm input output :class-name (or class-name "GeneratedDataset"))]
      (println "✓ Generated GSDM dataset:" output)
      (println "  Class:" (or class-name "GeneratedDataset"))
      (println "  Variables:" (:num-vars result))
      (println "  Edges:" (:num-edges result)))))
