// RUN: bishengir-opt %s -test-hivm-transform-patterns=test-hoist-affine -allow-unregistered-dialect -split-input-file | FileCheck %s

#map = affine_map<()[s0] -> (s0 + 1)>
#map1 = affine_map<()[s0, s1] -> (s0, s1)>
module {
  func.func @foo(%arg0: index) {
  // CHECK: affine.apply
  // CHECK: affine.min
  // CHECK: cf.br
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = affine.apply #map()[%arg0]
    %1 = affine.min #map1()[%arg0, %0]
    "some_use"(%1) : (index) -> ()
    return
  }
}

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  func.func @foo(%arg0: index, %arg1: index) {
  // CHECK: affine.apply
  // CHECK: cf.br
    cf.br ^bb1(%arg1: index)
  ^bb1(%0: index):
    %index = "some_op"(%0) : (index) -> (index)
    // CHECK: affine.apply
    // CHECK: "some_use"
    // CHECK: "some_use"
    %1 = affine.apply #map()[%index]
    %2 = affine.apply #map()[%arg0]
    "some_use"(%1) : (index) -> ()
    "some_use"(%2) : (index) -> ()
    return
  }
}

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  func.func @foo(%arg0: index, %arg1: index) {
    cf.br ^bb1(%arg0: index)
  ^bb1(%0: index):
     %index = "some_op"(%0) : (index) -> (index)
     // CHECK: affine.apply
     cf.br ^bb2(%arg1: index)
    ^bb2(%1: index):
      %2 = affine.apply #map()[%index]
      "some_use"(%2) : (index) -> ()
  return
  }
}

// -----

#map = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
module {
  func.func @foo(%arg0: index, %arg1: index) {
    cf.br ^bb1(%arg0: index)
  ^bb1(%0: index):
     %index = "some_op"(%0) : (index) -> (index)
     // CHECK: affine.apply
     cf.br ^bb2
    ^bb2():
      %2 = affine.apply #map()[%index, %arg1]
      "some_use"(%2) : (index) -> ()
  return
  }
}

// -----

#map = affine_map<()[s0] -> (s0 + 1)>
module {
  func.func @foo(%arg0: index) {
    cf.br ^bb1(%arg0: index)
  ^bb1(%0: index):
    %1 = "some_op"(%0) : (index) -> (index)
    // CHECK: affine.apply
    // CHECK: affine.apply
    %2 = affine.apply #map()[%1]
    "some_use"(%2) : (index) -> ()
    %3 = affine.apply #map()[%1]
    "some_use"(%3) : (index) -> ()
    return
  }
}