%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% *** Acknowledge ***
% The structure of my code draws on the structrue of the code named "sim_mm1.m" 
% in COMP9334 S1, 2017 WEEK6.
% lecture, source: https://webcms3.cse.unsw.edu.au/COMP9334/17s1/resources/7336
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load saved_rand_setting
rng(rand_setting)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Arrival rate
lambda = 7.2;
% the number of valid servers
s = 7;
% Simulation time
Tend = 2000;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Accounting parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 0; % T is the cumulative response time 
N = 0; % number of completed customers at the end of the simulation
%
% The mean response time will be given by T/N
% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialising the events
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Initialising the arrival event 
% 
next_arrival_time = (-log(1-rand(1))/lambda)* ((1.17-0.75)*rand()+0.75);
service_time_next_arrival = nthroot(1.289/rand(), 0.86) / (1.25+0.31*((2000/s)/200 - 1));

% Job list recorder
jobs = [next_arrival_time, service_time_next_arrival];

% 
% Initialise the departure event to empty
% Note: We use Inf (= infinity) to denote an empty departure event
% 
next_departure_time = Inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialising the Master clock, server status, queue_length,
% job_list
% 
% server_status = 1 if busy, 0 if idle
% 
% queue_length is the number of customers in the buffer
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Intialise the master clock 
master_clock = 0; 
% 
% Initialise Job List
job_list = [];
queue_length = 0;
% Initialise former_event_time
former_event_time = 0;
% Initialise partial_completion, for updating the job list
partial_completion = 0;
% Initialise rest_time, means the accumulated time with no jobs
rest_time = 0;
%

% Start iteration until the end time
while (master_clock < Tend)
    
    % Find out whether the next event is an arrival or depature
    %
    % We use next_event_type = 1 for arrival and 0 for departure
    % 
    if (next_arrival_time < next_departure_time)
        next_event_time = next_arrival_time;
        next_event_type = 1;  
    else
        next_event_time = next_departure_time;
        next_event_type = 0;
    end    
    
    %     
    % update master clock
    % 
    former_event_time = master_clock;
    master_clock = next_event_time;
        
    %
    % take actions depending on the event type
    % 
    if (next_event_type == 1) % an arrival 
        % 
        % add customer to job_list and
        % increment queue length
        % need to update:
        % (i) next arrival time
        % (ii) job list and (iii) next departure time
        %
        % update next arrival time
        %
        

        %
        % update job list
        %
        if queue_length
            partial_completion = (master_clock - former_event_time) / queue_length;
            job_list(:,2) = job_list(:,2) - partial_completion;
        end
        
        job_list = [job_list ; next_arrival_time service_time_next_arrival];
        queue_length = queue_length + 1;  
        
        % 
        % update next departure time
        %
        if ~(queue_length == 1)
            minimum_workload = Inf;
            for i=1:queue_length
                if minimum_workload > job_list(i,2)
                    % disp(job_list(i,2))
                    minimum_workload = job_list(i,2);
                end
                %disp(minimum_workload)
            end
            next_departure_time = master_clock + minimum_workload/((master_clock-former_event_time)/queue_length-1);
        else % the former is a departure with empty job list
            next_departure_time = master_clock + job_list(1,2);
            rest_time = master_clock - former_event_time; % no job time
        end
        % generate a new job and schedule its arrival 
        jump_job_list = 0;
        for i=1:s
            jump_job_list = jump_job_list + (-log(1-rand(1))/lambda)* ((1.17-0.75)*rand()+0.75);
        end
        next_arrival_time = master_clock + jump_job_list;
        for i=1:s
            service_time_next_arrival = (nthroot(1.289/rand(1,1), 0.86)) / (1.25+0.31*((2000/s)/200 - 1));
        end
        if next_arrival_time < Tend
            jobs = [jobs; next_arrival_time, service_time_next_arrival];
        end
    else % if it is a departure
        % 
        % Update the variables:
        % 1) Cumulative response time T
        % 2) Number of departed customers N
        % 3) job list
        % 4) next departure time
        %
        T = master_clock - rest_time; % rest_time is the time with no jobs
        %
        %
        % update job list
        % update N
        %
        if queue_length
            partial_completion = (master_clock - former_event_time) / queue_length;
            job_list(:,2) = job_list(:,2) - partial_completion;
            job_list_ = [];
            queue_length_ = 0;
            for i=1:queue_length
                %
                % ~(job_list(i,2) == 0) -> ~(job_list(i,2) < 0.0001)
                % because different degree of accuracy may cause infinite
                % loop, therefore, we assume a job which
                % left service time < 0.0001 can be regarded as a finished
                % job.
                % this part of the code would ensure validity when two or
                % more jobs finished at the same time
                %
                if ~(job_list(i,2) < 0.0001)
                    job_list_ = [job_list_;job_list(i,:)];
                    queue_length_ = queue_length_ + 1;
                end
            end
            job_list = job_list_;
            N = N + (queue_length - queue_length_);
            queue_length = queue_length_;
        end
            %
            % update next_departure_time
            %
        if queue_length
            almost_gone_job = Inf;
            for i=1:queue_length
                if almost_gone_job > job_list(i,2)
                    almost_gone_job = job_list(i,2);
                end
            end
            next_departure_time = master_clock + almost_gone_job/(1/queue_length);
        else
            next_departure_time = Inf;
        end
    end
    
end


% The estimated mean response time
disp(['s == ', num2str(s)])
disp(['The estimated mean response time is ',num2str(T/N)])

% producing job lists
%format long g
%disp('The total job list is: ')
%disp(jobs)

% vobtain setting and save it in a file
%rand_setting =rng;
%save saved_rand_setting rand_setting
%ps_server





