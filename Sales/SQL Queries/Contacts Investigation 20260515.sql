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

SELECT * from accounts.accounts_2026_05_15