- check word in corpus dict
  - if found -> put in bag_of_result
  - else
    - apply rule to current_words -> (rule.replacement / rule.replacements) -> put into current_words
      - each word in current_words
        - check word in corpus dict
          - if found -> put in bag_of_result
          
          - else
            - if recovery_mode == NO_RECOVERY -> take replacement put in current_words, remove original word in current_words (overwrite)
            - if recovery_mode == RECOVER -> revert back to original word in current_words
            - if recovery_mode == BOTH -> take replacement put in current_words and also revert back to original word in current_words


