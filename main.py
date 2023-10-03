from datetime import datetime as dt
from dateutil import tz
from datetime import timezone, timedelta

date = dt.now(tz=tz.gettz('Africa/Cairo'))
print(f'cairo time zone is: {date}')  # +02
# print(date.isoformat())

# ----------------------------------------------------------------------------
# EST = timezone(timedelta(hours=-5))
# EDT = timezone(timedelta(hours=-4))
#
# spring_ahead_159am = dt(2017, 3, 12, 6, 0, 0)
# spring_ahead_159am = spring_ahead_159am.replace(tzinfo=EST)
# spring_ahead_159am.isoformat()
#
# spring_ahead_3am = dt(2017, 3, 12, 7, 0, 0)
# spring_ahead_3am = spring_ahead_3am.replace(tzinfo=EDT)
# spring_ahead_3am.isoformat()
#
# print(spring_ahead_3am)  # 11
# print(spring_ahead_159am)  # 9
# print(spring_ahead_3am - spring_ahead_159am)
# print("")
# spring_ahead_159am = dt(2017, 3, 12, 9, 0, 0)
# spring_ahead_3am = dt(2017, 3, 12, 10, 0, 0)
# print(spring_ahead_3am - spring_ahead_159am)

# ----------------------------------------------------------------------------

# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo=tz.gettz('America/New_York'))
print(start)
print(timedelta(hours=6))
end = start + timedelta(hours=6)

print(f'before adding 6 hours: {start}')
print(f'after adding 6 hours: {end}')
print(end - start)
print(tz.datetime_ambiguous(end, tz=tz.gettz('America/New_York')))

print(start.tzinfo)
print(end.tzinfo)
print(end - start)
