function outfilts=prepare_dualfilters_2d(infilters)

fifi.psi{1} = infilters.psi;
fifi.phi{1} = infilters.phi;

lp = littlewood_paley_ISCV(fifi);
%keyboard
[dualpsi, dualphi]= dualwavelets(infilters.psi,infilters.phi,lp{1});

outfilts.psi = dualpsi;
outfilts.phi = dualphi;


