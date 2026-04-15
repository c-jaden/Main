-- create or replace temp table schools_priv as
SELECT 
name as "Account Name"
,phone as "Phone"
,coalesce(nullif(concat(LEFT(diocese_name, LEN(diocese_name) - 1),UPPER(RIGHT(diocese_name, 1))),''),'No District - Private School Placeholder') as "School District"
,website as "Website"
,'School' as "Account Type"
,address as "Billing Street"
,city as "Billing City"
,state_abbr as "Billing State"
,zip as "Billing Code"
,'United States' as "Billing Country"
,'Private' as "Type"
,county_name as "Billing County"
,cast(enrollment as decimal(38,0)) as "Students"
,grade_low as "Lowest Grade"
,grade_high as "Highest Grade"
,coalesce(diocese_code,'') as "NCES District ID"
,nces_id as "NCES School ID"

from nces_public_private_school
where school_type = 'private'
;

-- create or replace temp table dio_priv as
SELECT name as "Account Name"
,phone as "Phone"
,dio_website as "Website"
,'School District' as "Account Type"
,address as "Billing Street"
,city as "Billing City"
,primary_state as "Billing State"
,zip as "Billing Code"
,'United States' as "Billing Country"
,'Private' as "Type"
,cast(total_enrollment as decimal(38,0)) as "Students"
,grade_low as "Lowest Grade"
,grade_high as "Highest Grade"
,diocese_code as "NCES District ID"
,school_count as "Number of Schools"
from districts_dioceses
where diocese_code is not null
;

SELECT schools_priv.*
    ,dio_priv."Account Name"
from schools_priv
left join dio_priv
    on schools_priv."NCES District ID" = dio_priv."NCES District ID"