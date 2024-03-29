(defproject daphne "0.1.0-SNAPSHOT"
  :description "daphne is a probabilistic programming language compiler for
  programs that can be compiled to graphical models."
  :url "https://github.com/plai-group/daphne"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [org.clojure/core.memoize "1.0.257"]

                 [anglican "1.1.0"]
                 [aysylu/loom "1.0.2"]
                 [org.clojure/core.match "1.0.0"]
                 ;; for hy emission
                 [backtick "0.3.4"]
                 ;; for command line fu
                 [org.clojure/tools.cli "1.0.206"]
                 [camel-snake-kebab "0.4.3"]]

  :jvm-opts ["-Xss64m" "-Xmx4g"]

  :main daphne.command

  :min-lein-version "2.0.0")
