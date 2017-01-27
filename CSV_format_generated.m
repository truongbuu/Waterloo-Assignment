A = csvread('projectiles.csv');

B= A;
B(:,4) = B(:,2);
B(:,5) = B(:,3);
size_A = length(A);

new_traj = 0;
M =[]; 
j = 1;
for i = 1:size_A
    if A(i,1) == 0
       new_traj = 1; 
       M(j) = i;
       j = j+1;
    end
end

for i = 1: (length(M)-1)
    for j = M(i) : M(i+1)
        B(j,2) = B(M(i)+1,2);
        B(j,3) = B(M(i)+1,3);
        M(i)
    end
end
last_rows = M(i+1);

for j = last_rows : size_A
    B(j,2) = B(last_rows+1,2);
    B(j,3) = B(last_rows+1,3);
end
csvwrite('Data.csv',B);


vx = 10*sin(45*pi/180);
vy = 10*cos(45*pi/180);
t = [1:100]*0.1;
x = vx*t;
y = vy*t -0.5*9.8*t.^2;
A = [x;y];
csvwrite('Ground_truth.csv',A');

vx = 1000*sin(45*pi/180);
vy = 1000*cos(45*pi/180);
t = [1:2000]*0.1;
x = vx*t;
y = vy*t -0.5*9.8*t.^2;
A = [x;y];
csvwrite('Ground_truth_extreme.csv',A');
