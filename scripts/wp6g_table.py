import json, sys
J = json.load(open(sys.argv[1])); rows = J["rows"]
f = lambda r: f"{r['rate']:.4f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]" if r and r["n"] else "  n/a"
print(f"git {rows[0]['git_sha']}  enc {rows[0]['encoder']}  nli {rows[0]['nli_id']}  batch check {J['run']['batch_check']}")
print(f"{'dom':<9}{'a':>5}{'m':>3} {'mode':<7}{'arm':<4}{'FM tested [CI]':<24}{'k/n':>10}{'binmax(g)':>14}{'FMelig':>8}{'worstU':>8}{'recT':>7}{'tgtRec':>7}{'grpRec':>7}{'wrong':>6}{'unlM':>10}{'nli/d':>6}{'test/d':>7}  epoch")
for r in rows:
    fm, bm = r["false_merge"], r["fm_bin_max"]
    print(f"{r['domain']:<9}{r['alpha']:>5}{r['m']:>3} {r['mode']:<7}{r['arm']:<4}{f(fm):<24}{fm['k']:>5}/{fm['n']:<5}"
          f"{(f'{bm[chr(114)+chr(97)+chr(116)+chr(101)]:.4f}({bm[chr(98)+chr(105)+chr(110)]})' if bm else 'n/a'):>14}"
          f"{r['fm_eligible_population']['rate']:>8.4f}{r['fm_worst_unlabelled']['rate']:>8.4f}{r['merge_recall_tested']['rate']:>7.4f}"
          f"{r['target_recall']['rate']:>7.4f}{r['group_recall']['rate']:>7.4f}{r['wrong_target_merges']:>6}"
          f"{str(r['merges_into_unlabelled'])+'/'+str(r['unlabelled_tested']):>10}{r['nli_calls_per_delta']:>6.2f}{r['tested_per_delta']:>7.2f}  {r['epoch_id'][:8]}")
