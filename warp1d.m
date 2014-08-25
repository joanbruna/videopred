
function out=warp1d(in,field)

field=squeeze(field);
if(size(field,1) > size(field,2))
		field=field';
end
swap=0;
if(size(in,1) > size(in,2))
		in=in';
		swap=1;
end
x=[1:length(in)];
xx=min(length(in),max(1,x+field));
out=spline(x,[in(1) in in(end)],xx);

if swap
		out=out';
end

end

%function out=warp2d(in,field)
%
%%out(x)=in(x+field(x)) using bicubic interpolation
%
%intmp=in;
%fieldtmp=field;
%
%in=max(in(:))*ones(size(in,1)+2,size(in,2)+2);
%field=zeros(2,size(in,1),size(in,2));
%
%in(2:end-1,2:end-1)=intmp;
%field(:,2:end-1,2:end-1)=fieldtmp;
%
%
%[N,M]=size(in);
%
%for n=1:N
%		win(n,:)=warp1d(in(n,:),field(1,n,:) );
%end
%size(win)
%for m=1:M
%		out(:,m)=warp1d(win(:,m),field(2,:,m));
%end
%
%out=out(2:end-1,2:end-1);
%
%end
%
%
