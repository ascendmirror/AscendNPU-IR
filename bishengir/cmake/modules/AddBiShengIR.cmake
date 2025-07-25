# Declare the bishengir library associated with a conversion.
function(add_bishengir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_CONVERSION_LIBS ${name})
  add_mlir_dialect_library(${ARGV})
endfunction(add_bishengir_conversion_library)

# Declare the bishengir library associated with a dialect.
function(add_bishengir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_DIALECT_LIBS ${name})
  add_mlir_dialect_library(${ARGV})
endfunction(add_bishengir_dialect_library)