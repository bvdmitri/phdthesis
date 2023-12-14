@constraints begin 
  # The `..` iterates over all variables in the sequence `s`
  # `s[begin]` refers to the first element in the sequence
  # `s[end]` refers to the end element in the sequence
  q(s) = q(s[begin])..q(s[end]) 
end
