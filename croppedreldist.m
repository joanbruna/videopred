function out=croppedreldist(in, ref)

bord=3;

dife=in-ref;

tmp1=dife(bord:end-bord,bord:end-bord);
tmp2=ref(bord:end-bord,bord:end-bord);

out=norm(tmp1(:))/norm(tmp2(:));

