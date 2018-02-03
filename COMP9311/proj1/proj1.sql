-- COMP9311 16s1 Project 1
--
-- MyMyUNSW Solution Template


-- Q1: students who have taken more than 55 courses
create or replace view Q1(unswid, name)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q2: get details of the current Heads of Schools
create or replace view Q2(name, school, starting)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q3 UOC/ETFS ratio
create or replace view Q3(ratio,nsubjects)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q4: convenor for the most courses
create or replace view Q4(name, ncourses)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q5: program enrolments from 05S2
create or replace view Q5a(id)
as
--... SQL statements, possibly using other views/functions defined by you ...
;

create or replace view Q5b(id)
as
--... SQL statements, possibly using other views/functions defined by you ...
;

create or replace view Q5c(id)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q6: semester names
-- Testing case in check.sql: SELECT * FROM Q6(123);
create or replace function
	Q6(integer) returns text
as
$$
--... SQL statements, possibly using other views/functions defined by you ...
$$ language sql;



-- Q7: percentage of international students, S1 and S2, starting from 2005
create or replace view Q7(semester, percent)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q8: subjects with > 25 course offerings and no staff recorded
create or replace view Q8(subject, nOfferings)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q9: find a good research assistant
create or replace view Q9(unswid, name)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



-- Q10: find all students who had been enrolled in all popular subjects
create or replace view Q10(unswid, name)
as
--... SQL statements, possibly using other views/functions defined by you ...
;



