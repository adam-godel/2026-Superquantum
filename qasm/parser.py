import re
from pathlib import Path

GATE = re.compile(r'gate\s+(\w+)\s+([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\{(.*?)\}', re.S)
CALL = re.compile(r'^\s*(\w+)\s+([^;]+);\s*$')
QDECL = re.compile(r'^\s*qubit\[(\d+)\]\s+q\s*;\s*$', re.M)

def parse_ops(block):
    return [ln.strip() for ln in block.splitlines() if ln.strip() and ln.strip().endswith(";")]

def expand_line(line, env, gates, out):
    m = CALL.match(line)
    if not m: return
    name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]

    # map args through env (so gate-local names become q[i], etc.)
    args = [env.get(a, a) for a in args]

    # builtin op -> emit
    if name not in gates:
        out.append(f"{name} {', '.join(args)};")
        return

    # custom gate -> inline with new env
    formal_args, body_ops = gates[name]
    new_env = dict(env)
    for fa, aa in zip(formal_args, args):
        new_env[fa] = aa
    for op in body_ops:
        expand_line(op, new_env, gates, out)

def reformat(in_file):
    p = Path(in_file)
    s = p.read_text(encoding="utf-8")

    # collect includes (keep them)
    includes = "\n".join(re.findall(r'^\s*include\s+"[^"]+"\s*;\s*$', s, re.M))
    if not includes:
        includes = 'include "stdgates.inc";'

    # parse gates
    gates = {}
    for name, argstr, body in GATE.findall(s):
        gates[name] = ([a.strip() for a in argstr.split(",")], parse_ops(body))

    # find main program start: after qubit[...] q;
    qm = QDECL.search(s)
    if not qm:
        raise SystemExit("No 'qubit[n] q;' found.")
    n = int(qm.group(1))
    main = s[qm.end():]
    main_ops = parse_ops(main)

    # expand main ops
    out_ops = []
    for op in main_ops:
        expand_line(op, {}, gates, out_ops)

    # output name parsed_unitaryX.qasm next to input
    m = re.search(r'unitary(\d+)\.qasm$', p.name)
    X = m.group(1) if m else "0"
    out_path = p.with_name(f"parsed_unitary{X}.qasm")

    out_path.write_text(
        "OPENQASM 3.0;\n"
        f"{includes}\n"
        f"qubit[{n}] q;\n"
        + "\n".join(out_ops)
        + "\n",
        encoding="utf-8",
    )
    print(out_path)

if __name__ == "__main__":
    import sys
    reformat(sys.argv[1])
