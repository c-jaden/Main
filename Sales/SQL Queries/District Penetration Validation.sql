SELECT *
FROM accounts.accounts_2026_06_17 a
where 1=1
-- and a."Account Type" = 'School District'
and a."School District" = 'Racine Unified School District'
;

select "Squad Status" 
    ,count(*) record_count
    ,count("NCES School ID") / count(*) * 100 as pct_with_nces_id
from squads.squads_2026_06_17
-- where "Squad Status" in ('Active','In-Training')

group by 1