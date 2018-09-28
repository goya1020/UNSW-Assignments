
% Question 1
trigger([], goals([],[])).

trigger([restaurant(X,Y,S)|Tail1], goals([goal(X,Y,S)|Tail2], Truffles)) :-
		trigger(Tail1, goals(Tail2, Truffles)).

trigger([truffle(X,Y,S)|Tail1], goals(Restaurant, [goal(X,Y,S)|Tail2])) :-
		trigger(Tail1, goals(Restaurant, Tail2)).

% Question 2
incorporate_goals(goals([],[]),_,Intentions,Intentions).

% if the truffle is a new truffle
incorporate_goals(goals([],[goal(X,Y,S)|Goals_truff]), Beliefs, intents(Int_sell,Int_pick), Intentions1) :-
        incorporate_goals(goals([], Goals_truff), Beliefs, intents(Int_sell,Int_pick),intents(Int_sell1,Int_pick1)),
        not(member([goal(X,Y,S),_],Int_pick)),
        insert_new(goal(X,Y,S),Beliefs,Int_pick1,Int1Pick),
        Intentions1 = intents(Int_sell1,Int1Pick).

% if the truffle is not new
incorporate_goals(goals([],[goal(X,Y,S)|Goals_truff]), Beliefs, intents(Int_sell,Int_pick), Intentions1) :-
        incorporate_goals(goals([], Goals_truff), Beliefs, intents(Int_sell,Int_pick),Intentions2),
        member([goal(X,Y,S),_],Int_pick),
        Intentions1 = Intentions2.

% if the restaurant is a new restaurant
incorporate_goals(goals([goal(X,Y,S)|Goals_rest],Goals_truff), Beliefs, intents(Int_sell,Int_pick), Intentions1) :-
        not(member([goal(X,Y,S),_],Int_sell)),
        incorporate_goals(goals(Goals_rest,Goals_truff), Beliefs, intents(Int_sell,Int_pick), intents(Int_sell1,Int_pick1)),
        insert_new(goal(X,Y,S),Beliefs,Int_sell1,Int1Sell),
        Intentions1 = intents(Int1Sell,Int_pick1).

% if the restaurant is not new
incorporate_goals(goals([goal(X,Y,S)|Goals_rest],Goals_truff), Beliefs, intents(Int_sell,Int_pick), Intentions1) :-
        member([goal(X,Y,S),_],Int_sell),
        incorporate_goals(goals(Goals_rest,Goals_truff), Beliefs, intents(Int_sell,Int_pick), Intentions2),
        Intentions1 = Intentions2.


% insert when there is a new goal
insert_new(goal(X,Y,S),_,[],[[goal(X,Y,S),[]]]) :-!.

insert_new(goal(X1,Y1,S1),beliefs(at(X,Y),stock(T)),[[goal(X2,Y2,S2),Plan]|Rest_Intentions],[[goal(X2,Y2,S2),Plan]|Rest_Intentions1]) :-
        distance((X,Y),(X1,Y1),New_D),
        distance((X,Y),(X2,Y2),D),
        New_D > D,
        insert_new(goal(X1,Y1,S1),[at(X,Y)],Rest_Intentions,Rest_Intentions1),!.

insert_new(goal(X1,Y1,S1),beliefs(at(X,Y),stock(T)),[[goal(X2,Y2,S2),Plan]|Rest_Intentions],[[goal(X1,Y1,S1),[]]|Rest_Intentions1]) :-
        distance((X,Y),(X1,Y1),New_D),
        distance((X,Y),(X2,Y2),D),
        New_D < D,
        Rest_Intentions1 = [[goal(X2,Y2,S2),Plan]|Rest_Intentions],!.

insert_new(goal(X1,Y1,S1),beliefs(at(X,Y),stock(T)),[[goal(X2,Y2,S2),Plan]|Rest_Intentions],[[goal(X2,Y2,S2),Plan]|Rest_Intentions1]) :-
        distance((X,Y),(X1,Y1),New_D),
        distance((X,Y),(X2,Y2),D),
        New_D is D,
        S1 =< S2,
        insert_new(goal(X1,Y1,S1),[at(X,Y)],Rest_Intentions,Rest_Intentions1),!.

insert_new(goal(X1,Y1,S1),beliefs(at(X,Y),stock(T)),[[goal(X2,Y2,S2),Plan]|Rest_Intentions],[[goal(X1,Y1,S1),[]]|Rest_Intentions1]) :-
        distance((X,Y),(X1,Y1),New_D),
        distance((X,Y),(X2,Y2),D),
        New_D is D,
        S1 > S2,
        Rest_Intentions1 = [[goal(X2,Y2,S2),Plan]|Rest_Intentions],!.





