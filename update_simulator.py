# _schedule_prefill()와 _schedule_decode()를 찾아서 수정:

# Before:
# batch = self.scheduler.schedule_prefill_batch(self.current_time)

# After:
def memory_checker(requests, is_prefill):
    return self.memory_manager.can_schedule_batch(requests, is_prefill)

batch = self.scheduler.schedule_prefill_batch(
    self.current_time,
    memory_checker=memory_checker
)
