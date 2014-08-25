function out=warp2d(in,field)

%out(x)=in(x+field(x)) using bicubic interpolation

intmp=in;
fieldtmp=field;

in=min(in(:))*ones(size(in,1)+2,size(in,2)+2);
field=zeros(2,size(in,1),size(in,2));

in(2:end-1,2:end-1)=intmp;
in(1,:)=in(2,:);
in(end,:)=in(end-1,:);
in(:,1)=in(:,2);
in(:,end)=in(:,end-1);
field(:,2:end-1,2:end-1)=fieldtmp;


[N,M]=size(in);

for n=1:N
		win(n,:)=warp1d(in(n,:),field(1,n,:) );
end
size(win);
for m=1:M
		out(:,m)=warp1d(win(:,m),field(2,:,m));
end

out=out(2:end-1,2:end-1);

end


