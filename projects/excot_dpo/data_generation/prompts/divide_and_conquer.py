# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DIVIDE_AND_CONQUER_SYS = """
As a Text2SQL assistant, your main task is to formulate an SQL query in response to a given natural language inquiry. This process involves a chain-of-thought (CoT) approach, which includes a 'divide and conquer' strategy.

In the 'divide' phase of this CoT process, we break down the presented question into smaller, more manageable sub-problems using pseudo-SQL queries. During the 'conquer' phase, we aggregate the solutions of these sub-problems to form the final response.

Lastly, we refine the constructed query in the optimization step, eliminating any unnecessary clauses and conditions to ensure efficiency.

Here is the template:

Database Info
..DATABASE_SCHEMA..
**************************
Question
Question: ..question..
**************************
## Divide and Conquer

### Main Question:  ..main question..
**Analysis:**
..analysis..

```Pseudo sql
..pseudo sql..
```

### Sub-question 1: ..sub question..
**Analysis:**
..analysis..

```Pseudo sql
..pseudo sql..
```

### Sub-question 1.1: ..sub question..
**Analysis:**
..analysis..

```Pseudo sql
..pseudo sql..
```

...

## Assembling SQL

### Sub-question 1.1: sub-question 1.1

```sql
..sql..
```

### Sub-question 1: sub-question 1

```sql
..sql..
```

### Main Question: main question

```sql
..sql..
```

### Simplification and Optimization

**Analysis:**
..analysis..

```sql
..sql..
```
""".strip()


few_shot_0 = """
Database Info
taughtby :
course_id [ NUMBER ] ( "0" , "1" , "2" ) taughtby.course_id=course.course_id
p_id [ NUMBER ] ( "40" , "279" , "107" ) primary_key taughtby.p_id=person.p_id

person :
hasposition [ TEXT ] ( "0" , "Faculty" , "Faculty_adj" )
yearsinprogram [ TEXT ] ( "0" , "Year_2" , "Year_5" )
p_id [ NUMBER ] ( "3" , "4" , "5" ) primary_key
inphase [ TEXT ] ( "0" , "Post_Quals" , "Post_Generals" )
professor [ NUMBER ] ( "0" , "1" )
student [ NUMBER ] ( "1" , "0" )

advisedby :
p_id [ NUMBER ] ( "96" , "118" , "75" ) advisedby.p_id=person.p_id
p_id_dummy [ NUMBER ] ( "5" , "7" , "29" ) primary_key advisedby.p_id_dummy=person.p_id

course :
courselevel [ TEXT ] ( "Level_500" , "Level_300" , "Level_400" )
course_id [ NUMBER ] ( "0" , "1" , "2" ) primary_key
**************************
Question
Question: List down all the person IDs who taught course ID of 18.
**************************
""".strip()

few_shot_0_output = """
## Divide and Conquer

### Main Question: List down all the person IDs who taught course ID of 18
**Analysis:**
To address the question of identifying all person IDs who taught `course ID = 18`, we need to concentrate on the `taughtby` table where the relationship between `person` and `course` is maintained.
The `course_id` column should be used to find `course_id = 18`, and we aim to return `p_id` for person IDs. Understanding the table we need to consider, we can break the question down into selecting the person id from a table. This table should be filtered using the taught course ID of 18.

```Pseudo sql
SELECT p_id FROM taughtby with course_id = 18;
```

### Sub-question 1: Filter the table to get rows with taught course ID of 18
**Analysis:**
In table `taughtby`, there is a column `course_id`, and we are looking to retrieve all records where `course_id = 18`.

```Pseudo sql
SELECT * FROM taughtby WHERE course_id = 18;
```


## Assembling SQL
### Sub-question 1: Filter the table to get rows with taught course ID of 18

```sql
SELECT * FROM taughtby WHERE course_id = 18;
```

### Main Question: Identify Records which is the person IDs from the table

```sql
SELECT p_id
FROM (SELECT *
      FROM taughtby
      WHERE course_id = 18) AS intermediate_table;
```


### Simplification and Optimization

**Analysis:**
The nested queries can be simplified without using an intermediate table

```sql
SELECT p_id FROM taughtby WHERE course_id = 18;
```
""".strip()

few_shot_2 = """
Database Info
advisedby :
p_id_dummy [ NUMBER ] ( "5" , "7" , "29" ) primary_key advisedby.p_id_dummy=person.p_id
p_id [ NUMBER ] ( "96" , "118" , "75" ) advisedby.p_id=person.p_id

course :
course_id [ NUMBER ] ( "0" , "1" , "2" ) primary_key
courselevel [ TEXT ] ( "Level_500" , "Level_300" , "Level_400" )

taughtby :
course_id [ NUMBER ] ( "0" , "1" , "2" ) taughtby.course_id=course.course_id
p_id [ NUMBER ] ( "40" , "279" , "107" ) primary_key taughtby.p_id=person.p_id

person :
yearsinprogram [ TEXT ] ( "0" , "Year_2" , "Year_5" )
p_id [ NUMBER ] ( "3" , "4" , "5" ) primary_key
inphase [ TEXT ] ( "0" , "Post_Quals" , "Post_Generals" )
student [ NUMBER ] ( "1" , "0" )
hasposition [ TEXT ] ( "0" , "Faculty" , "Faculty_adj" )
professor [ NUMBER ] ( "0" , "1" )
**************************
Question
Question: What is the average number of courses taught by a professor?
**************************
""".strip()

few_shot_2_output = """
## Divide and Conquer

### Main Question: What is the average number of courses taught by a professor?
**Analysis:**
The average number of courses taught by a professor is calculated by dividing the total number of courses by the total number of professors. To achieve this:
- The `person` table contains an indicator column `professor` that identifies the professors.
- The `taughtby` table has a `course_id` associated with each `p_id`.
- The `p_id` serves as a common key between the `person` and `taughtby` tables.
By joining these two tables on `p_id`, filtering for professors (`professor = 1`), and counting the `course_id`, we can then divide by the distinct number of `p_id` to compute the average number of courses taught by a professor.

```pseudo sql
SELECT CAST(COUNT(taughtby.course_id) AS FLOAT) / COUNT(DISTINCT person.p_id) AS num # average calculation
FROM taughtby table join with person table with professor = 1 # table join and filter
```
### Sub-question 1: Join the `taughtby` table with the `person` table with `professor=1`
**Analysis:**
To link the two tables, we join them using the `p_id` column.

```pseudo sql
SELECT *
FROM taughtby
INNER JOIN person with professor = 1
ON taughtby.p_id = person.p_id;
```
### Sub-question 1.1: Identify professors
**Analysis:**
To isolate professors, we need to filter the `person` table where `professor = 1`.

```pseudo sql
SELECT *
FROM person
WHERE professor = 1;
```


## Assembling SQL
### Sub-question 1.1: Identify professors

```sql
SELECT *
FROM person
WHERE professor = 1;
```
### Sub-question 1: Join the `taughtby` table with the `person` table with `professor=1`

```sql
SELECT *
FROM taughtby
INNER JOIN (SELECT * FROM person WHERE professor = 1) AS prof
ON taughtby.p_id = prof.p_id;
```
### Main Question: Calculate the average number of courses taught by a professor

```sql
SELECT CAST(COUNT(taughtby.course_id) AS FLOAT) / COUNT(DISTINCT person.p_id) AS num
FROM taughtby
INNER JOIN (SELECT * FROM person WHERE professor = 1) AS prof
ON taughtby.p_id = prof.p_id;
```
## Simplification and Optimization
### Analysis:
We can streamline the SQL query by combining the filtering for professors and the join into a single step, reducing complexity.
### Final Optimized SQL Query:

```sql
SELECT CAST(COUNT(taughtby.course_id) AS FLOAT) / COUNT(DISTINCT person.p_id) AS num
FROM taughtby
INNER JOIN person ON taughtby.p_id = person.p_id
WHERE person.professor = 1;
""".strip()


few_shot_3 = """
Database Info
country :
country_name [ TEXT ] ( "United Arab Emirates" , "Afghanistan" , "Angola" )
country_iso_code [ TEXT ] ( "AE" , "AF" , "BG" )
country_id [ NUMBER ] ( "128" , "129" , "130" ) primary_key

keyword :
keyword_id [ NUMBER ] ( "30" , "65" , "74" ) primary_key
keyword_name [ TEXT ] ( "individual" , "holiday" , "germany" )

department :
department_name [ TEXT ] ( "Camera" , "Directing" , "Production" )
department_id [ NUMBER ] ( "1" , "2" , "3" ) primary_key

production_company :
company_name [ TEXT ] ( "Lucasfilm" , "Walt Disney Pictures" , "Paramount Pictures" )
company_id [ NUMBER ] ( "1" , "2" , "3" ) primary_key

language_role :
role_id [ NUMBER ] ( "1" , "2" ) primary_key
language_role [ TEXT ] ( "Original" , "Spoken" )

genre :
genre_name [ TEXT ] ( "Adventure" , "Fantasy" , "Animation" )
genre_id [ NUMBER ] ( "12" , "14" , "16" ) primary_key

movie_company :
company_id [ NUMBER ] ( "14" , "59" , "1" ) movie_company.company_id=production_company.company_id
movie_id [ NUMBER ] ( "5" , "11" , "12" ) movie_company.movie_id=movie.movie_id

gender :
gender_id [ NUMBER ] ( "0" , "1" , "2" ) primary_key
gender [ TEXT ] ( "Unspecified" , "Female" , "Male" )

language :
language_code [ TEXT ] ( "en" , "sv" , "de" )
language_name [ TEXT ] ( "English" , "svenska" , "Deutsch" )
language_id [ NUMBER ] ( "24574" , "24575" , "24576" ) primary_key

movie :
movie_status [ TEXT ] ( "Released" , "Rumored" , "Post Production" )
movie_id [ NUMBER ] ( "5" , "11" , "12" ) primary_key
homepage [ TEXT ] ( "" , "http://www.starwars.com/films/star-wars-episode-iv-a-new-hope" , "http://movies.disney.com/finding-nemo" )
overview [ TEXT ] ( "It's Ted the Bellhop's first night on the job...and the hotel's very unusual guests are about to place him in some outrageous predicaments. It seems that this evening's room service is serving up one unbelievable happening after another." , "Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire. Venturesome Luke Skywalker and dashing captain Han Solo team together with the loveable robot duo R2-D2 and C-3PO to rescue the beauti" , "Nemo, an adventurous young clownfish, is unexpectedly taken from his Great Barrier Reef home to a dentist's office aquarium. It's up to his worrisome father Marlin and a friendly but forgetful fish Dory to bring Nemo home -- meeting vegetarian sharks, sur" )
revenue [ NUMBER ] ( "4300000" , "775398007" , "940335536" )
release_date [ TEXT ] ( "1995-12-09" , "1977-05-25" , "2003-05-30" )
popularity [ FLOAT ] ( "22.87623" , "126.393695" , "85.688789" )
budget [ NUMBER ] ( "4000000" , "11000000" , "94000000" )
title [ TEXT ] ( "Four Rooms" , "Star Wars" , "Finding Nemo" )
runtime [ NUMBER ] ( "98" , "121" , "100" )
**************************
Question
Question: Calculate the revenues made by Fantasy Films and Live Entertainment.
**************************
""".strip()


few_shot_3_output = """
## Divide and Conquer

### Main Question: Calculate the revenues made by Fantasy Films and Live Entertainment, two companies.
**Analysis:**
The question outlines the need to check the gross revenues for two movie company, Fantasy Films and Live Entertainment. To achieve this:
- The `movie` table contains the `revenue` to calculate the gross revenues asked in the main question.
- The `production_company` table has `company_name` we need to get the two company we want, and it is associated with each `company_id`.
- The `movie_company` include both `movie_id` and `company_id` which can build bridge with two tables with join operations on `movie_id` and `company_id`
- The names of two companies have been filtered, so we're unable to directly use the 'where' clause. Given that the number of companies is not too large, using the 'IN' operator would be a suitable choice.

```pseudo sql
SELECT SUM(movie.revenue) FROM
A set of JOIN to get the company movie mappings, and rows' company_name IN ('Fantasy Films', 'Live Entertainment')
```

### Sub-question 1: Join the `production_company` table with the `movie` table
**Analysis:**
`production_company` has `company_name` and `company_id`, and table `movie` has `movie_id` as primary key, but don't have common keys with `production_company`, we can use `movie_company` which include both `movie_id` and `company_id`, we can join them. After joining, because `movie_id` and `company_id` are both primary key so they are unique in both `production_company` and `movie`, so we don't need worried duplication issue.

```pseudo sql
production_company INNER JOIN movie_company ON company_id INNER JOIN movie ON movie_id
```

### Sub-question 2: Identify company names from `production_company`.
**Analysis:**
We have two company names, usually we use WHERE to filter, but we have two names, so use IN is better fit.

```pseudo sql
SELECT ... production_company WHERE company_name IN ('Fantasy Films', 'Live Entertainment');
```

## Assembling SQL

### Sub-question 2: Identify company names from `production_company`

```sql
SELECT * FROM production_company WHERE production_company.company_name IN ('Fantasy Films', 'Live Entertainment')
```

### Sub-question 1: Join the `production_company` table with the `movie` table

```sql
SELECT * FROM production_company INNER JOIN movie_company ON production_company.company_id = movie_company.company_id INNER JOIN movie ON movie_company.movie_id = movie.movie_id WHERE production_company.company_name IN ('Fantasy Films', 'Live Entertainment')
```

### Main Question: Calculate the revenues made by Fantasy Films and Live Entertainment, two companies

```sql
SELECT SUM(movie.revenue) FROM production_company INNER JOIN movie_company ON production_company.company_id = movie_company.company_id INNER JOIN movie ON movie_company.movie_id = movie.movie_id WHERE production_company.company_name IN ('Fantasy Films', 'Live Entertainment')
```

### Simplification and Optimization

**Analysis:**
The sql code in Sub-question 3 is clean and efficent enough, can pass to final directly.

```sql
SELECT SUM(movie.revenue) FROM production_company INNER JOIN movie_company ON production_company.company_id = movie_company.company_id INNER JOIN movie ON movie_company.movie_id = movie.movie_id WHERE production_company.company_name IN ('Fantasy Films', 'Live Entertainment')
```
""".strip()

messages = [
    {"role": "system", "content": DIVIDE_AND_CONQUER_SYS},
    {"role": "user", "content": few_shot_0},
    {"role": "assistant", "content": few_shot_0_output},
    {"role": "user", "content": few_shot_2},
    {"role": "assistant", "content": few_shot_2_output},
    {"role": "user", "content": few_shot_3},
    {"role": "assistant", "content": few_shot_3_output},
]
