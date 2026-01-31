OPENQASM 3.0;
include "stdgates.inc";
gate _circuit_43 _gate_q_0 {
  s _gate_q_0;
  s _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  s _gate_q_0;
  h _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  t _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
}
gate _circuit_42 _gate_q_0 {
  s _gate_q_0;
  _circuit_43 _gate_q_0;
  sdg _gate_q_0;
}
gate _circuit_49 _gate_q_0 {
  sdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  sdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  sdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  sdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  sdg _gate_q_0;
  h _gate_q_0;
  tdg _gate_q_0;
  sdg _gate_q_0;
  h _gate_q_0;
  h _gate_q_0;
  s _gate_q_0;
  s _gate_q_0;
  h _gate_q_0;
  sdg _gate_q_0;
  sdg _gate_q_0;
}
gate _circuit_48 _gate_q_0 {
  s _gate_q_0;
  _circuit_49 _gate_q_0;
  sdg _gate_q_0;
}
qubit[2] q;
_circuit_42 q[1];
cx q[0], q[1];
_circuit_48 q[1];
cx q[0], q[1];
