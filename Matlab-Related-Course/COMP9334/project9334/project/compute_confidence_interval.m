disp('s6 means the system of 6 opening servers.')
disp('s7 means the system of 7 opening servers.')
s6 = [0.5179 0.51923 0.51734 0.51576 0.52083 0.51653 0.51641 0.52083 0.5169 0.51784];
s7 =[0.51472 0.51585 0.51527 0.51633 0.51489 0.51431 0.51633 0.51575 0.51597 0.51695];
s76 = s7 - s6;

n = 10;
sample_mean = sum(s6)/10;
s6_ = sample_mean-s6;
s6_ = s6_.^2;
sample_standard_deviation = nthroot(sum(s6_)/(n-1), 2);
alpha = 0.05;
t = tinv(1-alpha/2,n-1);
s6_lower = sample_mean - t*(sample_standard_deviation/nthroot(n,2));
s6_upper = sample_mean + t*(sample_standard_deviation/nthroot(n,2));

disp(['s6 confidence_interval is ', num2str(s6_lower),' to ', num2str(s6_upper)])


sample_mean = sum(s7)/10;
s7_ = sample_mean-s7;
s7_ = s7_.^2;
sample_standard_deviation = nthroot(sum(s7_)/(n-1), 2);
s7_lower = sample_mean - t*(sample_standard_deviation/nthroot(n,2));
s7_upper = sample_mean + t*(sample_standard_deviation/nthroot(n,2));

disp(['s7 confidence_interval is ', num2str(s7_lower),' to ', num2str(s7_upper)])


sample_mean = sum(s76)/10;
s76_ = sample_mean-s76;
s76_ = s76_.^2;
sample_standard_deviation = nthroot(sum(s76_)/(n-1), 2);
s76_lower = sample_mean - t*(sample_standard_deviation/nthroot(n,2));
s76_upper = sample_mean + t*(sample_standard_deviation/nthroot(n,2));

disp(['confidence_interval of difference between s6 and s7 is ', num2str(s76_lower),' to ', num2str(s76_upper)])
disp('Therefore, s7 is better than s6.')