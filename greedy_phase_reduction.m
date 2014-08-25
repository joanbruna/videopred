function Pout = greedy_phase_reduction(Pin, SAT)

Pin = 2*Pin - 1;

marginals = abs(SAT)'*abs(Pin);

[~, Ind] = sort(abs(Pin),1,'ascend'); 
%[~, MInd] = sort(marginals,1,'ascend'); 

[M, N]=size(SAT);
[M, L]=size(Pin);

maxits=2;

Pout = sign(randn(N, L));

pair=zeros(M,2);
for m=1:M
pair(m,:)=find(SAT(m,:)~=0);
end

aux=M*[0:L-1];
aux2=N*[0:L-1];

for it=1:maxits
fprintf('current assignment has %f \n', evaluate_assignment(Pout, Pin, pair))
for m=1:M
tmp = Ind(m,:) + aux;
chunk = sign(Pin(tmp));
%satisfy constraints in chunk
tmp2 = pair(Ind(m,:),1);
%keyboard
%size(tmp2)
T1 = pair(Ind(m,:),1)'+aux2;
T2 = pair(Ind(m,:),2)'+aux2;
val1 = Pout(T1);
val2 = Pout(T2);
%marg1 = marginals(T1);
%marg2 = marginals(T2);

%1st case: chunk==1 and T1==T2: agreement

%2nd case: chunk==1 and T1~=T2: we need to flip one bit
mm = find((chunk==1).*(val1~=val2));
Pout(T2(mm)) = Pout(T1(mm));

%3rd case: chunk~=1 and T1==T2: we need to flip one bit
mm = find((chunk~=1).*(val1==val2));
Pout(T2(mm)) =- Pout(T1(mm));

%4th case: chunk~=1 and T1~=T2: agreement

end
end

end

function out=evaluate_assignment(P, Pin, pair)

%P is the input phase vector
%Pin contains the priors on the phase differences
%[N, L]=size(P);
%[M, L]=size(Pin);

p1=pair(:,1);
p2=pair(:,2);

P1 = P(p1,:);
P2 = P(p2,:);

C = P1 .* P2;

out=sum(C(:).*Pin(:))/numel(Pin);


end




