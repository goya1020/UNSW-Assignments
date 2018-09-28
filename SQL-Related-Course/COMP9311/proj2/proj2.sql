--Q1
CREATE TYPE TranscriptRecord AS (code CHAR(8), term CHAR(4), course INTEGER, prog CHAR(4), name TEXT, mark INTEGER, grade CHAR(2), uoc INTEGER, rank INTEGER, totalEnrols INTEGER);


CREATE TABLE ranktable AS
  SELECT
    course,
    student,
    cast(rank()
         OVER (PARTITION BY course
           ORDER BY mark DESC) AS INTEGER) AS rank
  FROM course_enrolments
  WHERE mark IS NOT NULL;


CREATE OR REPLACE VIEW q6_a(result, id)
AS
  SELECT
    (cast(year AS VARCHAR(4)) || term) AS raw_answer,
    id
  FROM semesters;

CREATE OR REPLACE VIEW q6_b(result, id2)
AS
  SELECT
    lower(result),
    id
  FROM q6_a;

CREATE OR REPLACE FUNCTION
  semester(INTEGER)
  RETURNS TEXT
AS
$$
SELECT substr($1, 3)
FROM q6_b
WHERE q6_b.id2 = $1
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION unswid_to_id(INTEGER)
  RETURNS INT
AS $$
SELECT id
FROM people
WHERE unswid = $1
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION one_rank(INTEGER, INTEGER)
  RETURNS INTEGER
AS $$
DECLARE
  inter INTEGER;
BEGIN
  SELECT rank
  INTO inter
  FROM ranktable
  WHERE student = $1 AND course = $2;
  RETURN inter;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION how_many_people_in_a_course(INTEGER)
  RETURNS SETOF INTEGER
AS $$
BEGIN
  RETURN QUERY SELECT cast(count(student) AS INTEGER)
               FROM course_enrolments
               WHERE course = $1 AND mark IS NOT NULL
               GROUP BY course;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION check_uoc(CHAR(2), INTEGER, INTEGER)
  RETURNS INTEGER AS $$
DECLARE
  grade ALIAS FOR $1;
  uoc ALIAS FOR $2;
  mark ALIAS FOR $3;
BEGIN
  IF (grade not in ('SY', 'PC', 'PS', 'CR', 'DN', 'HD', 'PT', 'A', 'B')) AND (mark IS NOT NULL)
  THEN
    RETURN 0;
  ELSE
    RETURN uoc;
  END IF;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION Q1(INTEGER)
  RETURNS SETOF TranscriptRecord
AS $$
BEGIN
  RETURN QUERY SELECT DISTINCT
                 subjects.code,
                 cast(semester(courses.semester) AS CHARACTER(4))  AS term,
                 course,
                 programs.code                                     AS prog,
                 cast(subjects.name AS TEXT),
                 mark,
                 cast(grade AS CHARACTER(2)),
                 check_uoc(course_enrolments.grade, subjects.uoc, mark),
                 one_rank(course_enrolments.student, course) AS rank,
                 how_many_people_in_a_course(course)                         AS totalEnrols
               FROM course_enrolments
                 LEFT JOIN courses ON course_enrolments.course = courses.id
                 LEFT JOIN subjects ON courses.subject = subjects.id
                 LEFT JOIN program_enrolments ON (course_enrolments.student = program_enrolments.student) AND
                                                 (courses.semester = program_enrolments.semester)
                 LEFT JOIN programs ON program_enrolments.program = programs.id
               WHERE course_enrolments.student = (SELECT *
                                                  FROM unswid_to_id($1));
END
$$ LANGUAGE plpgsql;




-- Q2
CREATE TYPE MatchingRecord AS ("table" TEXT, "column" TEXT, nexamples INTEGER);


CREATE OR REPLACE FUNCTION columns(TEXT)
  RETURNS SETOF TEXT
AS $$
DECLARE
BEGIN
  RETURN QUERY SELECT cast(a.attname AS TEXT) AS name
               FROM pg_class AS c, pg_attribute AS a
               WHERE c.relname = $1 AND a.attrelid = c.oid AND a.attnum > 0;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION check_pattern(xcolumn TEXT, pattern TEXT, xtable TEXT)
  RETURNS INTEGER
AS $$
DECLARE
  n INTEGER;
BEGIN
  FOR n IN EXECUTE 'SELECT COUNT(*) FROM (SELECT REGEXP_MATCHES('|| $1 ||', $2) FROM '|| $3 ||') AS nexamples'
  USING xcolumn, pattern, xtable LOOP
    RETURN n;
  END LOOP;
  EXCEPTION
  WHEN SQLSTATE '42883'
    THEN RETURN NULL;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION Q2("table" TEXT, pattern TEXT)
  RETURNS SETOF MatchingRecord
AS $$
DECLARE
  current_column TEXT;
  nexamples      INTEGER := 0;
BEGIN
  FOR current_column IN SELECT *
                        FROM columns($1) LOOP
    IF check_pattern(current_column, pattern, $1) <> 0 AND
       check_pattern(cast(current_column AS TEXT), pattern, $1) IS NOT NULL
    THEN
      nexamples := check_pattern(current_column, pattern, $1);
    END IF;
    IF nexamples <> 0
    THEN
      RETURN QUERY SELECT
                     $1,
                     current_column,
                     nexamples;
    END IF;
    nexamples := 0;
  END LOOP;
END;
$$ LANGUAGE plpgsql;




-- Q3
CREATE OR REPLACE FUNCTION people_in(INTEGER)
  RETURNS TABLE(staff_id INTEGER, orgunit INTEGER, role INTEGER, isprimary BOOLEAN, starting DATE, ending DATE)
AS $$
BEGIN
  RETURN QUERY SELECT *
               FROM affiliations
               WHERE affiliations.orgunit IN (SELECT member
                                              FROM orgunit_groups
                                              WHERE owner = $1
                                              UNION SELECT $1)
               ORDER BY staff;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION to_initial(INTEGER)
  RETURNS TABLE(staff INTEGER, orgunit INTEGER, role INTEGER, isprimary BOOLEAN, starting DATE, ending DATE)
AS $$
BEGIN
  RETURN QUERY SELECT *
               FROM affiliations
               WHERE affiliations.staff IN (SELECT staff_id
                                            FROM people_in($1)
                                            GROUP BY staff_id
                                            HAVING count(staff_id) > 1) AND
                     affiliations.staff IN (SELECT DISTINCT affiliations.staff
                                            FROM affiliations
                                            WHERE affiliations.staff IN (SELECT staff_id
                                                                         FROM people_in($1)
                                                                         GROUP BY staff_id
                                                                         HAVING count(staff_id) > 1) AND
                                                  affiliations.ending IS NOT NULL)
                                                  AND affiliations.orgunit IN (SELECT member
                                              FROM orgunit_groups
                                              WHERE owner = $1
                                              UNION SELECT $1)
               ORDER BY staff;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION further_choose_persons(INTEGER)
  RETURNS SETOF INTEGER
AS $$
BEGIN
  RETURN QUERY SELECT DISTINCT staff
               FROM affiliations
               WHERE staff IN (SELECT staff_id
                               FROM people_in($1)
                               GROUP BY staff_id
                               HAVING count(staff_id) > 1) AND ending IS NOT NULL;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION all_startings(orgs INTEGER, staf INTEGER)
  RETURNS SETOF DATE
AS $$
BEGIN
  RETURN QUERY SELECT starting
               FROM to_initial(orgs)
               WHERE staff = staf;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION all_endings(orgs INTEGER, staf INTEGER)
  RETURNS SETOF DATE
AS $$
BEGIN
  RETURN QUERY SELECT ending
               FROM to_initial(orgs)
               WHERE staff = staf;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION finally_check(orgs INTEGER)
  RETURNS SETOF INTEGER
AS $$
DECLARE
  person   INTEGER;
  xend   DATE;
  xstart DATE;
BEGIN
  FOR person IN SELECT *
                FROM further_choose_persons(orgs) LOOP
    FOR xend IN SELECT *
                  FROM all_endings(orgs, person) LOOP
      FOR xstart IN SELECT *
                      FROM all_startings(orgs, person) LOOP
        IF xend <= xstart
        THEN RETURN NEXT person;
        END IF;
      END LOOP;
    END LOOP;
  END LOOP;
END
$$ LANGUAGE plpgsql;

CREATE TYPE EmploymentRecord AS (unswid INTEGER, name TEXT, roles TEXT);

CREATE OR REPLACE FUNCTION comments(INTEGER)
  RETURNS TABLE(unswid INTEGER, name TEXT, roles TEXT, starttt DATE, sortnameee TEXT)
AS $$
DECLARE
  xid       INTEGER;
  nid          INTEGER;
  xname          TEXT;
  xroles         TEXT;
  xorgunits         TEXT;
  xstarting         DATE;
  xending      DATE;
  order_starting    DATE;
  order_sortname TEXT;
BEGIN
  FOR xid IN SELECT *
              FROM finally_check($1) LOOP
    FOR nid, xname, xroles, xorgunits, xstarting, xending, order_starting, order_sortname IN SELECT
                                                          people.unswid,
                                                          people.name,
                                                          staff_roles.name,
                                                          orgunits.name,
                                                          affiliations.starting,
                                                          affiliations.ending,
                                                          affiliations.starting,
                                                          people.sortname
                                                        FROM affiliations
                                                          JOIN staff_roles ON role = staff_roles.id
                                                          JOIN orgunits ON orgunit = orgunits.id
                                                          JOIN people ON people.id = xid
                                                        WHERE staff = xid and affiliations.orgunit IN (SELECT member
                                                            FROM orgunit_groups
                                                            WHERE owner = $1
                                                            UNION SELECT $1) LOOP
      IF xending IS NOT NULL
      THEN RETURN QUERY SELECT
                          nid,
                          xname,
                          xroles || ', ' || xorgunits || ' (' || xstarting || '..' || xending || ')',
                          order_starting,
                          order_sortname;
      ELSE RETURN QUERY SELECT
                          nid,
                          xname,
                          xroles || ', ' || xorgunits || ' (' || xstarting || '..)',
                          order_starting,
                          order_sortname;
      END IF;
    END LOOP;
  END LOOP;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION check_people2(INTEGER)
  RETURNS SETOF INTEGER
AS $$
BEGIN
  RETURN QUERY SELECT DISTINCT people.unswid
               FROM people
               WHERE people.id IN (SELECT *
                                   FROM further_choose_persons($1));
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION draft(INTEGER)
  RETURNS SETOF EmploymentRecord
AS $$
DECLARE
  xunswid INTEGER;
BEGIN
  FOR xunswid IN SELECT *
                  FROM check_people2($1) LOOP
    RETURN QUERY SELECT
                   foo.unswid,
                   foo.name,
                   foo.roles
                 FROM (SELECT DISTINCT
                         unswid,
                         name,
                         roles,
                         starttt
                       FROM comments($1)) AS foo
                 WHERE xunswid = foo.unswid
                 ORDER BY starttt;
  END LOOP;
END
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION _Q3(INTEGER)
  RETURNS SETOF EmploymentRecord
AS $$
DECLARE
  i        INTEGER;
  xunswid INTEGER;
  xname   TEXT;
  xroles  TEXT;
BEGIN
  FOR i IN SELECT DISTINCT unswid
           FROM draft($1) LOOP
    FOR xunswid, xname, xroles IN SELECT distinct
                                       unswid,
                                       name,
                                       array_to_string(array(SELECT roles
                                                             FROM draft($1)
                                                             WHERE unswid = i), E'\n')
                                     FROM draft($1)
                                     WHERE unswid = i LOOP
      xroles := xroles || E'\n';
      RETURN QUERY SELECT
                     xunswid,
                     xname,
                     xroles;
    END LOOP;
  END LOOP;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION Q3(INTEGER)
  RETURNS SETOF EmploymentRecord
AS $$
BEGIN
  return query select
                 r.unswid,
                 r.name,
                 r.roles
               from _Q3($1) as r
                 join people on r.unswid = people.unswid
               order by sortname;
END
$$ LANGUAGE plpgsql;