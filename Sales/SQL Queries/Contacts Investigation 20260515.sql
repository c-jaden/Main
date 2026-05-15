select c."Record Id"
    ,a.account_id
    ,a."Account Name"
from contacts.contacts_2026_05_15 c
left join accounts.accounts_2026_05_15 a
    on a."NCES District ID" = c."NCES District ID"
    and a."Account Type" = 'School District'
where c."Tag" = 'Purchase List - March 26'
and a."NCES District ID" is not null
;

SELECT * from contacts.contacts_2026_05_15
where "First Name" = 'Bethany'
and "Last Name" = 'Henry'
;

select c."Record Id"
    ,a.account_id
    ,a."Account Name"
from contacts.contacts_2026_05_15 c
left join accounts.accounts_2026_05_15 a
    on a."NCES School ID" = c."NCES School ID"
    and a."Account Type" = 'School'
where c."Tag" = 'Purchase List - March 26'
and c."Title" = 'Public Principal'
and c."NCES School ID" is not null
;

select * from contacts.contacts_2026_05_15 c
left join accounts.accounts_2026_05_15 d
    on a."NCES District ID" = c."NCES District ID"
    and a."Account Type" = 'School District'
left join accounts.accounts_2026_05_15 s
    on a."NCES School ID" = c."NCES School ID"
    and a."Account Type" = 'School'
where c."Tag" = 'Purchase List - March 26'


limit 100
;
wide net May 2026
;

with district_squads as (
    select a.account_id as district_id
        ,sc.account_id as school_id
        ,count(*) as squad_cnt
    from squads.squads_2026_05_15 s
    left join accounts.accounts_2026_05_15 sc
        on sc."Record Id" = s."School.id"
        and sc."Account Type" = 'School'
    left join accounts.accounts_2026_05_15 a
        on a."Record Id" = sc."School District.id"
        and a."Account Type" = 'School District'

    group by 1,2
)
select c."Record Id"
    ,'Wide Net - May 2026' as "Tag"
from contacts.contacts_2026_05_15 c
left join district_squads ss
    on ss.school_id = c.account_id
left join district_squads ds
    on ds.district_id = c.account_id
where c."Tag" = 'Purchase List - March 26'

group by 1

having sum(coalesce(ss.squad_cnt,ds.squad_cnt,0)) = 0
;

select  from squads.squads_2026_05_15