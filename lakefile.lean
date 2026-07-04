import Lake
open Lake DSL

package "SCPNControl"

@[default_target]
lean_lib SCPNControl where
  srcDir := "lean"
