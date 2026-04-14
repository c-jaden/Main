create or replace temp table private_acct as
SELECT 
name as "Account Name"
,phone as "Phone"
,concat(LEFT(diocese_name, LEN(diocese_name) - 1),UPPER(RIGHT(diocese_name, 1))) as "School District"
,website as "Website"
,'School' as "Account Type"
,city as "Billing City"
,state_abbr as "Billing State"
,zip as "Billing Code"
,'United States' as "Billing Country"
,'Private' as "Type"
,county_name as "Billing County"
,cast(enrollment as decimal(38,0)) as "Students"
,grade_low as "Lowest Grade"
,grade_high as "Highest Grade"
,diocese_code as "NCES District ID"
,nces_id as "NCES School ID"

from nces_public_private_school
where school_type = 'private'
;
